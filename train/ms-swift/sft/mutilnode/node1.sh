nnodes=2
nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=$nnodes \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /data/models/Qwen/Qwen3-235B-A22B \
    --train_type lora \
    --dataset /root/workspace/swift/datasets/SFT_finetune_data_more.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $(expr 32 / $nproc_per_node / $nnodes) \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --deepspeed zero2 \
    --output_dir output \
    --report_to swanlab \
    --swanlab_project swift_sft_lora \
    --swanlab_token token \
    --swanlab_exp_name zero2_Qwen3-8B \
    --swanlab_mode cloud \