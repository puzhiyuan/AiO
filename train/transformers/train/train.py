import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import TrainingArguments
import numpy as np


# 将数据集转换为 Qwen3 template格式
# Qwen3 的 template 格式是的一个最小的对话格式，例如：
# [
#     {"role": "user", "content": "你好"},
#     {"role": "assistant", "content": "你好，有什么可以帮助你的吗？"}
# ]
def convert_to_qwen3_format(user_content, assistant_content):
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# 定义一个函数来处理数据集中的每个样本
def tokenize_function(example):
    # 检查 prompt 和 response 是否为字符串或列表(datasets.map 中 batch=True 时会将 prompt 和 response 转换为列表)
    if isinstance(example["prompt"], str) and isinstance(example["response"], str):
        qwen3_format_list = convert_to_qwen3_format(
            example["prompt"], example["response"]
        )

    if isinstance(example["prompt"], list) and isinstance(example["response"], list):
        qwen3_format_list = [
            convert_to_qwen3_format(prompt, response)
            for prompt, response in zip(example["prompt"], example["response"])
        ]

    text = tokenizer.apply_chat_template(
        qwen3_format_list, tokenize=False, add_generation_prompt=False
    )
    # # 使用 tokenizer 编码, 不进行填充,后面会在构建训练 batch 时进行填充
    # # 这里的 max_length 可以根据需要调整
    # # 注意：如果文本长度超过 max_length，tokenizer 会截断文本
    return tokenizer(text, truncation=True, max_length=1024)


def compute_metrics(eval_preds):
    # eval_preds 是一个 EvalPrediction 对象，包含 predictions 和 label_ids
    # predictions 是模型的原始输出（logits），一个元组，我们通常关心第一个元素
    # label_ids 是真实的标签
    logits, labels = eval_preds

    # logits 的形状是 (batch_size, sequence_length, vocab_size)
    # labels 的形状是 (batch_size, sequence_length)

    # 我们需要从 logits 中计算出预测的 token ID
    # np.argmax 会在最后一个维度（vocab_size）上找到最大值的索引
    predictions = np.argmax(logits, axis=-1)

    # 为了计算准确率，我们只关心那些非-100的标签（-100是DataCollator忽略的部分，如prompt和padding）
    # 创建一个布尔掩码，True 表示我们关心该位置的 token
    mask = labels != -100

    # 计算准确率
    # (predictions == labels)[mask] 会选出所有非忽略位置上预测正确的 token
    accuracy = np.sum((predictions == labels)[mask]) / np.sum(mask)

    # Trainer 在评估时会自动计算 loss，我们可以直接从 trainer.evaluate() 的结果中获取
    # 但我们也可以在这里手动计算，虽然没必要。
    # Perplexity 是交叉熵损失的指数。Trainer会返回'eval_loss'，所以我们可以在外面计算
    # 这里我们只返回 accuracy
    return {"accuracy": accuracy}


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        # device_map="cuda",  # 使用 ddp 时，不要指定 device_map
    )

    if tokenizer.pad_token is None:
        # 如果没有 pad_token，则设置为 eos_token
        tokenizer.pad_token = tokenizer.eos_token

    datasets = load_dataset("Mxode/Chinese-Instruct", "dpsk-r1-distil")
    split_datasets = datasets["train"].train_test_split(test_size=0.3, seed=42)

    # 使用 map 函数对数据集进行处理
    # 会将 tokenize_function 返回的结果添加到数据集的字段中
    split_datasets = split_datasets.map(
        tokenize_function,
        batched=True,  # 批处理模式
        remove_columns=split_datasets[
            "train"
        ].column_names,  # 移除原有的列，只保留 tokenized 的结果
        num_proc=16,  # 并行处理的进程数
        desc="Tokenizing dataset",
    )
    print(">>> Tokenization complete. \n Sample data:", split_datasets["train"][0])
    print(
        "Sample data decoded:",
        tokenizer.decode(split_datasets["train"][0]["input_ids"]),
    )
    print("sample data keys:", split_datasets["test"][0].keys())
    print("Train dataset size:", len(split_datasets["train"]))
    print("Test dataset size:", len(split_datasets["test"]))

    # exit(0)
    # 需要传入 tokenizer 来让其知道填充的 token、 padding的方式
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     pad_to_multiple_of=8,
    # )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 设置训练参数
    output_dir = "/root/zhiyuan/workspace/AiO/train/transformers/train/qwen3-sft-chinese-instruct"  # 模型保存的目录
    training_args = TrainingArguments(
        output_dir=output_dir,  # 模型保存的目录
        num_train_epochs=10,  # 训练的轮数
        learning_rate=2e-5,  # 学习率
        warmup_steps=100,  # 预热步数
        weight_decay=0.1,  # 权重衰减
        per_device_train_batch_size=32,  # 训练时的 batch size
        per_device_eval_batch_size=32,  # 评估时的 batch size
        gradient_checkpointing=False,  # DeepSpeed 本身也能管理梯度检查点，可以先不开启
        gradient_accumulation_steps=8,  # 梯度累积步数
        logging_dir=os.path.join(output_dir, "logs"),  # 日志目录
        eval_strategy="steps",  # 告诉 Trainer 在每 eval_steps 步后进行评估
        eval_steps=5,  # 每 5 步进行一次评估
        logging_strategy="steps",  # 同样按步数记录日志
        logging_steps=2,  # 每 2 步记录一次训练日志
        bf16=True,  # 或 fp16=True, 看你的GPU支持
        report_to="swanlab",  # 使用 SwanLab 进行日志记录
        dataloader_num_workers=4,  # 数据加载的工作线程数
        run_name="qwen3-finetuned-chinese-instruct",  # 训练任务的名称(在 SwanLab 中显示)
        save_total_limit=3,  # 最多保存3个模型
        save_steps=200,  # 每200步保存一次模型
        # ddp_find_unused_parameters=False,  # 如果使用 DDP 分布式训练，设置为 False, 使用 DeepSpeed 时, 移除
        # deepspeed="/root/zhiyuan/workspace/AiO/train/transformers/train/ds_config.json",
    )

    # 使用 Trainer 进行训练
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_datasets["train"],
        eval_dataset=split_datasets["test"],  # 使用分割后的测试集进行评估
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # 计算评估指标
    )

    train_results = trainer.train()  # 开始训练
    print(train_results)
    trainer.log_metrics("train", train_results.metrics)

    # 保存模型和 tokenizer
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
