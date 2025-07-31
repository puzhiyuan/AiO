CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model Kwai-Keye/Keye-VL-8B-Preview \
    --model_type keye_vl \
    --train_type lora \
    --dataset '/data2/zhiyuan/workspace/ms-swift/datasets/data/processed_nothink/train-00001-of-00001.parquet' \
              '/data2/zhiyuan/workspace/ms-swift/datasets/data/processed_think/train-00001-of-00001.parquet' \
              '/data2/zhiyuan/datasets/DJY/sft/merged_all_qa_noTelLatLong_extend_withThinkTagInInstruction_extendByCoTData.json' \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --eval_steps 50 \
    --save_steps 150 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 3096 \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 8 \
    --output_dir /data/models/swift_results/keye_vl_8b \
    --save_only_model true \
    --report_to swanlab \
    --swanlab_project swift_sft_lora_vlm \
    --swanlab_exp_name sft-djy-mm-full-params \

# tips:
# AttributeError: Can't pickle local object 'DeepSpeedEngine._create_module_forward_post_hook.<locals>._module_forward_post_hook'
# use "deepspeed<0.17" | --dataloader_num_workers 0

# models:
# model: /data/models/huggingface/Qwen/Qwen2.5-VL-7B-Instruct
# model_type: qwen2_5_vl

# model: Kwai-Keye/Keye-VL-8B-Preview
# model_type: keye_vl

# datasets:
# djy-sft: /root/zhiyuan/datasets/DJY/sft/merged_all_qa_noTelLatLong_extend_withThinkTagInInstruction_extendByCoTData.json
# djy-vl: /root/zhiyuan/workspace/ms-swift/datasets/data