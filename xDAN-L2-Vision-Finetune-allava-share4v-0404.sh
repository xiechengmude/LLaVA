#!/bin/bash
{
accelerate launch  llava/train/train_mem.py \
    --deepspeed ./deepspeed/zero3_offload.json \
    --model_name_or_path xDAN2099/xDAN-L2-RL-Mix378-BagelMath-0310-e2-Chat-v7.2-DPO-QDora-0318-epoch08 \
    --version chatml_direct \
    --data_path /workspace/LLaVA-HR/playground/data/LLaVA-Instruct-share4v-allava-cleaned-collections.json \
    --image_folder /workspace/LLaVA-HR/playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /workspace/LLaVA-HR/playground/models/projector/xDAN-L2-Vision-34b-v2-Pretrain-Projector-ckp7520/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./playground/checkpoints/xDAN-L2-Vision-RL-v7.2-e08-projector-ckp7520-Finetune-Allava-Shareg4v \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb
} > Task_Vision_Finetune_xDAN-L2-Vision-RL-Chat-v7.2-DPO-Allava-Sharegpt4v_$(TZ=Asia/Shanghai date +'%Y%m%d-%H-%M-%S').log 2>&1 &



#chatml_direct
#v1