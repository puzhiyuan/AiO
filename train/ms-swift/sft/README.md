# SFT（Supervised Fine-Tuning）脚本说明

本目录包含基于 [ModelSwift](https://github.com/modelscope/swift) 框架编写的监督微调（SFT）阶段训练脚本，支持全参数微调（Full）、参数高效微调（LoRA）以及多节点分布式训练（Multinode）。

## 📁 目录结构

```
sft/
├── full/                        # 全参数微调脚本
│   ├── train_ddp_device_map.sh  # 使用 device_map 的 DDP 训练脚本
│   ├── train_ddp.sh             # 标准 DDP 训练脚本
│   ├── train_zero2.sh           # ZeRO Stage 2 分布式训练
│   └── train_zero3.sh           # ZeRO Stage 3 分布式训练
├── lora/                        # LoRA 微调与合并脚本
│   ├── merge.sh                 # 合并 LoRA adapter 权重到原始模型
│   ├── train_ddp_device_map.sh  # 使用 device_map 的 DDP 训练
│   ├── train_ddp.sh             # 标准 DDP 训练脚本
│   ├── train_zero2.sh           # ZeRO Stage 2 微调
│   └── train_zero3.sh           # ZeRO Stage 3 微调
├── mutilnode/                   # 多节点训练脚本与通信配置
│   ├── env.sh                   # 分布式环境变量配置
│   ├── node1.sh                 # 第一台节点训练脚本
│   └── node2.sh                 # 第二台节点训练脚本
└── usage.txt                    # swift sft -h 的 CLI 参数说明
```



## 🔧 子模块说明

### 1. `full/` - 全参数微调

适用于显存资源充足的环境，对模型所有参数进行训练更新，提升效果但资源开销较大。

* `train_ddp.sh`：使用 PyTorch Distributed Data Parallel（DDP）方式进行单机多卡训练。
* `train_ddp_device_map.sh`：结合 HuggingFace 的 `device_map` 控制模型切片与 DDP 并行。
* `train_zero2.sh`：使用 DeepSpeed ZeRO Stage 2 分布式训练策略，适合中等规模模型。
* `train_zero3.sh`：使用 ZeRO Stage 3，进一步节省显存，适合大模型训练。



### 2. `lora/` - 参数高效微调（LoRA）

适用于资源受限场景，仅训练少量新增参数（Adapter），可大幅降低显存与训练成本。（但是效果可能不太好）

* `train_ddp.sh`：LoRA 模式下的标准 DDP 微调。
* `train_ddp_device_map.sh`：结合 `device_map` 与 DDP 的 LoRA 微调。
* `train_zero2.sh` / `train_zero3.sh`：结合 DeepSpeed ZeRO 的高效分布式训练。
* `merge.sh`：在训练完成后将 LoRA 权重合并到原始模型权重中，方便后续推理或部署。



### 3. `mutilnode/` - 多节点分布式训练

适用于跨节点的集群训练部署。

* `env.sh`：配置 NCCL 通信、主机 IP、端口等环境变量。
* `node1.sh` / `node2.sh`：分别为主节点与工作节点启动训练进程脚本。需根据实际机器 IP 与节点数修改。

详细参数可参考 `usage.txt`。
