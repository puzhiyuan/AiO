# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /root/zhiyuan/workspace/swift/sft/output/v0-20250610-115540/checkpoint-1160 \
    --merge_lora true \
    --output_dir /root/zhiyuan/workspace/swift/sft/output/v0-20250610-115540/merged-1160 \
