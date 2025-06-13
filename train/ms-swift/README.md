# AiO Training Scripts (Swift-based)

整理了基于 [ModelSwift](https://github.com/modelscope/swift) 的完整训练脚本流程，涵盖大模型的预训练（pretrain）、监督微调（SFT）和强化学习微调（RLHF，包括 GRPO 与 PPO）。

## 📁 项目结构

```
.
├── pretrain                     # 预训练脚本（使用 zero2 并行）
│   └── train_zero2.sh
├── rlhf                         # 强化学习阶段训练脚本
│   ├── grpo                     # GRPO算法训练脚本
│   │   ├── external_rollout     # 外部模型 rollout + 训练
│   │   └── internal_rollout     # 内部模型 rollout + 训练
│   └── ppo                      # PPO训练脚本
│       └── ppo_train.sh
├── sft                          # 监督微调脚本
│   ├── full                     # 全参数微调（全量训练）
│   ├── lora                     # LoRA 微调 + 模型合并
│   ├── mutilnode                # 多节点分布式训练（包含通信配置）
│   └── usage.txt                # swift sft -h 的命令参数说明
```

## 🧪 模块说明

### 1. `pretrain/`

用于模型的预训练流程。当前包含一个基于 ZeRO Stage 2 的训练脚本 `train_zero2.sh`，适用于大模型多卡训练。

请参见 [`pretrain/README.md`](./pretrain/README.md) 获取参数配置、模型结构与训练命令详细说明。


### 2. `sft/`

监督微调阶段，包含以下几种训练模式：

* `full/`：全量参数训练（包含 DDP 和 ZeRO Stage 2/3）
* `lora/`：参数高效训练（LoRA），支持训练与合并权重
* `mutilnode/`：支持多节点多卡训练，需配置通信环境

`usage.txt` 文件为 `swift sft -h` 的参数帮助文档，便于快速了解支持的 CLI 参数配置。


### 3. `rlhf/`

强化学习微调（RLHF）模块，包含两种训练策略：

* `grpo/`：包括内部与外部模型的 rollout 与训练脚本
* `ppo/`：标准 PPO 算法训练流程

推荐首先查看 `GRPO_From_Scratch_Multi_GPU_DataParallel_Qwen_2_5_1_5B_Instruct.ipynb`，理解整体训练流程。


## ⚙️ ms-swift 通用参数说明

| 参数                                                  | 含义                          |
| --------------------------------------------------- | --------------------------- |
| `--model`                                           | 指定微调基础模型路径                  |
| `--train_type`                                      | 训练模式：`full`（全参数）或 `lora`    |
| `--dataset`                                         | 输入数据集路径（支持 json/jsonl）      |
| `--max_length`                                      | 最大序列长度，影响显存与训练效率            |
| `--torch_dtype`                                     | 精度类型，如 `bfloat16`、`fp16`    |
| `--attn_impl`                                       | 注意力实现方式，推荐使用 `flash_attn`   |
| `--num_train_epochs`                                | 训练轮数                        |
| `--learning_rate`                                   | 学习率（如 1e-4）                 |
| `--warmup_ratio`                                    | 热身阶段占比                      |
| `--per_device_train_batch_size`                     | 每张卡的训练 batch size           |
| `--gradient_accumulation_steps`                     | 梯度累积步数，决定全局 batch size      |
| `--gradient_checkpointing_kwargs`                   | 显存优化设置                      |
| `--save_steps` / `--eval_steps` / `--logging_steps` | 日志与保存控制                     |
| `--output_dir`                                      | 模型输出目录                      |
| `--report_to`                                       | 日志平台，支持 `wandb`、`swanlab`   |
| `--swanlab_*`                                       | Swanlab 配置，包含 token、项目名、模式等 |
