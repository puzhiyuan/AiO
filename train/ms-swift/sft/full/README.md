# 🚀 Full SFT 全参数微调指南

在使用 `--train_type full` 对 Qwen3-8B 等大模型进行全参数微调时，需格外关注显存分布、优化策略、训练配置等关键维度。本指南提供常用参数说明、训练脚本范例，以及显存优化建议，帮助你高效开展 full SFT 训练任务。



## ⚙️ 关键参数详解

### `--train_type full`

开启全参数微调（Full SFT），训练时模型所有参数均可更新。相比 LoRA 插件更精细、学习能力更强，但显存占用显著增大。

适合用于：

* 自有场景高保真定制；
* 数据量充足、标签准确；
* 拥有高性能 GPU 资源（推荐 A100/H100）。



### `--torch_dtype bfloat16`

使用 `bfloat16` 精度进行训练：

* **优点**：几乎不损失精度情况下大幅减少显存占用；
* **要求**：GPU 需支持（如 A100、H100）；
* **默认推荐**：强烈建议开启。



### `--attn_impl flash_attn`

开启 FlashAttention 加速：

* 使用 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 优化注意力计算；
* 需提前安装插件库，显著提升大序列推理/训练性能；
* 建议在具备高性能 GPU 的环境中使用（如 A100）。



### `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`

激活梯度检查点以节省显存，同时关闭 `use_reentrant` 以避免与部分优化策略冲突（如 DDP）：

* 推荐与 `ddp`/`deepspeed` 配合使用；
* 会牺牲部分训练速度以换取显存空间。



### `--gradient_accumulation_steps`

控制梯度累积步数：

* 模拟大 Batch Size，适合大模型训练；
* 建议配合较小的 `per_device_train_batch_size`；
* 提升训练稳定性。



## 🧪 脚本说明与使用

### `train_ddp_device_map.sh`

使用 **DDP + 模型切片加载（device\_map）** 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset train_dataset.json \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 4 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --output_dir output \
    --report_to swanlab \
    --swanlab_project swift_sft_full \
    --swanlab_token token \
    --swanlab_exp_name ddp-mp_Qwen3-8B \
    --swanlab_mode cloud \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' 
```

**说明**：

* `NPROC_PER_NODE=4`：每个节点启动 4 个训练进程，建议每个进程绑定 2 张 GPU（形成 mp）。
* 建议与模型并行结合，以支持 8B+ 模型加载。



### `train_zero2.sh`

使用 **DeepSpeed ZeRO Stage 2** 进行训练，显著优化显存开销：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset train_dataset.json \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 4 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --deepspeed zero2 \
    --output_dir output \
    --report_to swanlab \
    --swanlab_project swift_sft_full \
    --swanlab_token token \
    --swanlab_exp_name zero2_Qwen3-8B \
    --swanlab_mode cloud
```

**注意事项**：

* `--deepspeed zero2` 会自动启用 ZeRO Stage 2，优化 optimizer state 和梯度存储；
* 若遇 OOM，可降低 `per_device_train_batch_size` 或提升 `gradient_accumulation_steps`。


## 🧠 小贴士：资源优化建议

* 显存不足时优先考虑：

  * 开启 `gradient_checkpointing`；
  * 启用 `--deepspeed zero2`；
  * 调整 `batch_size` + `gradient_accumulation_steps`；
  * 降低 `eval_steps` 或关闭 eval；
* 模型并行建议使用 `NPROC_PER_NODE=4` 配合 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 启动；
* 建议使用环境变量 `WANDB_MODE=offline` 或 `swanlab_mode=cloud` 来控制训练日志同步策略。
