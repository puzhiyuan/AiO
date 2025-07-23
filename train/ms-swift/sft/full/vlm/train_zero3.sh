CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model Kwai-Keye/Keye-VL-8B-Preview \
    --model_type keye_vl \
    --train_type full \
    --dataset '/data2/zhiyuan/workspace/ms-swift/datasets/data/processed_nothink/train-00001-of-00001.parquet' \
              '/data2/zhiyuan/workspace/ms-swift/datasets/data/processed_think/train-00001-of-00001.parquet' \
              '/data2/zhiyuan/datasets/DJY/sft/merged_all_qa_noTelLatLong_extend_withThinkTagInInstruction_extendByCoTData.json' \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --eval_steps 50 \
    --save_steps 200 \
    --save_total_limit 20 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 8 \
    --output_dir /data2/models/swift_results/keye_vl_8b \
    --save_only_model true \
    --report_to swanlab \
    --swanlab_project swift_sft_full_vlm \
    --swanlab_exp_name sft-djy-mm-full-params \

# tips:
# AttributeError: Can't pickle local object 'DeepSpeedEngine._create_module_forward_post_hook.<locals>._module_forward_post_hook'
# use "deepspeed<0.17" | --dataloader_num_workers 0

# 训练过程出现 _REDUCE_SCATTER_BASE 操作超时了，导致整个训练被强制中止
# 训练 vlm 模型时，数据中混合了纯文本数据和图像文本数据，解除 vit 和 aligner 的冻结可能会解决这个问题

# models:
# model: /data/models/huggingface/Qwen/Qwen2.5-VL-7B-Instruct
# model_type: qwen2_5_vl

# model: Kwai-Keye/Keye-VL-8B-Preview
# model_type: keye_vl

# datasets: 16852 + 264 + 260
# djy-llm-sft: /data2/zhiyuan/datasets/DJY/sft/merged_all_qa_noTelLatLong_extend_withThinkTagInInstruction_extendByCoTData.json
# djy-vl-nothink: /data2/zhiyuan/workspace/ms-swift/datasets/data/processed_nothink/train-00001-of-00001.parquet
# djy-vl-think: /data2/zhiyuan/workspace/ms-swift/datasets/data/processed_think/train-00001-of-00001.parquet