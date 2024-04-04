#!/bin/bash
{
accelerate launch  llava/train/train_mem.py \
    --deepspeed ./deepspeed/zero3_offload.json \
    --model_name_or_path liuhaotian/llava-v1.6-34b \
    --version chatml_direct \
    --data_path /workspace/LLaVA-HR/playground/data/LLaVA-Instruct-share4v-allava-cleaned-collections.json \
    --image_folder /workspace/LLaVA-HR/playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_projector_lr 2e-5 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type spatial_unpad \
    --image_aspect_ratio anyres \
    --group_by_modality_length False \
    --bf16 True \
    --fp16 False \
    --output_dir ./llava-lora-34b \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 
} > Task_Vision_Finetune_xDAN-L2-Nours-Allava-Sharegpt4v_$(TZ=Asia/Shanghai date +'%Y%m%d-%H-%M-%S').log 2>&1 &
