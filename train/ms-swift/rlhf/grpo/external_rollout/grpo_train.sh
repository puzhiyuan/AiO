CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_API_KEY=877cc2bba3285f3cb46a1dd39eb04f5ee841cc82 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model  /data/models/huggingface/Qwen/Qwen3-8B \
    --external_plugins /root/workspace/swift/ms-swift/examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_countdown format \
    --use_vllm true \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'zouxuhong/Countdown-Tasks-3to4#50000' \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --eval_steps 500 \
    --save_steps 100 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --output_dir output/xxx \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --deepspeed zero3 \
    --log_completions true \
    --beta 0.001 \
    --num_iterations 1 \
    --report_to swanlab \
    --swanlab_project swift_rlhf_grpo \
    --swanlab_token token \
    --swanlab_exp_name zero3_Qwen3-8B \
    --swanlab_mode cloud \