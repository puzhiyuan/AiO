CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model /root/zhiyuan/models/Qwen3-8B \
    --train_type full \
    --dataset /root/zhiyuan/datasets/DJY/sft/train_data.json \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
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
    --swanlab_exp_name ddp_Qwen3-8B \
    --swanlab_mode cloud \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' 
