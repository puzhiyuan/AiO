CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift pt \
    --model /data/models/huggingface/Qwen/Qwen3-8B \
    --train_type full \
    --dataset /root/zhiyuan/datasets/DJY/pt/pretrain_dataset.json \
    --torch_dtype bfloat16 \
    --streaming true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --packing true \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --output_dir output/xxx \
    --report_to swanlab \
    --swanlab_project swift_pt_full \
    --swanlab_token token \
    --swanlab_exp_name zero3_Qwen3-8B \
    --swanlab_mode cloud \