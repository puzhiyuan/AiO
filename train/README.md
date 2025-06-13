## 通用参数

### 1. 环境变量 (Environment Variables)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
```

* **作用**: 这个变量用来控制程序能“看到”并使用的 NVIDIA GPU。
* **`0,1,2,3,4,5,6,7`**: 表示这台服务器上编号为 0 到 7 的所有 8 个 GPU 对这个程序都是可见的。程序可以自由选择其中的任何一个。

```bash
NPROC_PER_NODE=4 \
```

* **`NPROC_PER_NODE`**: (Number of Processes Per Node) 这个变量通常被分布式训练启动器使用。
* **`4`**: 它告诉启动器在这台机器（Node）上启动 **4个** 训练进程。




DDP（Distributed Data Parallel）只做数据并行，每个进程有完整模型副本；
DDP + device_map（Model Parallel，例如 TP/PP）是更复杂的并行训练策略，将模型切分到多个设备/进程中，以解决模型太大无法放进单张显卡的问题。
