## ⚙️ LoRA 参数详解

在使用 `--train_type lora` 进行微调时，以下参数对于效果与资源消耗有显著影响：

### `--target_modules`

指定哪些模块应用 LoRA 插件：

* **推荐设置**：

  * Qwen 模型推荐使用注意力模块：`'[q_proj,k_proj,v_proj,o_proj]'`
  * 类似 LLaMA 或 GPT 模型也多推荐使用上述注意力模块。
* **默认值**：`all-linear`（作用于所有线性层）
* **建议**：

  * 若资源有限，可仅作用于 `q_proj` 和 `v_proj`。
  * 对于任务泛化性要求高，可尝试增加 `gate_proj` 或 `mlp` 层线性模块。

### `--lora_rank`

LoRA 插件的秩（rank）决定了引入的低秩矩阵容量大小：

* **含义**：控制 A × B 矩阵的秩（压缩程度）。
* **推荐范围**：4～64
* **默认值**：无固定默认，常见为 `8`。
* **越大越精细**，但显存占用、训练参数也越多。


### `--lora_alpha`

LoRA 的缩放比例参数：

* **含义**：在合成 A × B 的输出时，使用 `alpha / r` 作为缩放因子。
* **推荐范围**：16～64
* **默认值**：32 常见
* **建议**：
  * 一般配合 `lora_rank`，满足 `alpha / rank` ≈ 4 或 8；
  * 提高 `alpha` 可增强 LoRA 插件对模型输出的影响。


## 🧪 脚本说明与使用

### `train_ddp_device_map.sh`

使用 DDP 配合 `device_map` 切片加载模型：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    ...
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' 
```

**关键点：**

* `NPROC_PER_NODE=4` 表示使用 4 个进程并行训练，搭配 8 张显卡。（实现四组 dp，每组 dp 的两个模型上实现 mp）
* `--target_modules all-linear` 是对全模型线性层添加 LoRA 插件，推荐针对不同模型进行调整（如 `q_proj,v_proj`）。
* `--gradient_checkpointing_kwargs` 显式关闭 reentrant，避免与 LoRA 的梯度回传冲突。



### `train_zero2.sh`

基于 DeepSpeed ZeRO Stage 2 策略进行显存优化，适合中大型模型：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --deepspeed zero2 \
    ...
```

**注意：**

* `--deepspeed zero2` 会自动加载默认的 ZeRO 配置文件。
* 若遇 GPU 显存溢出，可适当调小 `--per_device_train_batch_size` 和 `--gradient_accumulation_steps`。



### `merge.sh`

用于将训练后的 LoRA adapter 权重合并到原始模型中（以便推理使用）：

```bash
swift export \
    --adapters /path/to/output/checkpoint-xxx \
    --merge_lora true \
    --output_dir /path/to/merged_model
```



## ⚠️ LoRA 微调中的注意事项

在使用 LoRA 进行大模型微调时，请特别注意以下几点：

1. **target_modules 指定模块**

   * 并非所有层都适合加 LoRA，推荐仅作用于注意力部分（如 Q、K、V、Out）。
   * 示例：`--target_modules '[q_proj,k_proj,v_proj]'`。使用 `all-linear` 会覆盖全部线性层，显存更高。
2. **bfloat16 与 flash attention**

   * 推荐开启 `--torch_dtype bfloat16`，可大幅降低显存占用。
   * `--attn_impl flash_attn` 需安装 [FlashAttention](https://github.com/Dao-AILab/flash-attention)，建议使用 A100/H100 等 GPU。
3. **梯度检查点（Gradient Checkpointing）**

   * 使用 ddp 训练，一定要关闭 `use_reentrant` ，否则可能会报错：

     ```bash
     --gradient_checkpointing_kwargs '{"use_reentrant": false}'
     ```
4. **微调效率**

   * 推荐通过 `--gradient_accumulation_steps` 模拟大 batch，同时确保显存可承受。
   * 设置合适的 `--save_steps` 与 `--eval_steps`，避免频繁 I/O。
5. **权重保存与推理**

   * 训练完成后，使用 `merge.sh` 合并权重才能进行无依赖推理部署。
   * 合并后的模型可直接用于 HuggingFace 推理接口（如 `AutoModelForCausalLM.from_pretrained()`）。

## 📌 小贴士

* 若模型显存吃紧，优先尝试：

  * 减少 `target_modules` 范围（仅保留注意力模块）；
  * 降低 `lora_rank`；
  * 增大 `gradient_accumulation_steps`；
  * 启用 `--deepspeed zero2`；
