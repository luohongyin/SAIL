#!/bin/bash


accelerate launch fastchat/train/train.py \
    --model_name_or_path $LLAMA_PATH  \
    --data_path data/SAIL_train.json \
    --fp16 True \
    --output_dir sail-lm \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --report_to none \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --fsdp "full_shard auto_wrap" \
     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 1600 \
    --gradient_checkpointing True \
    --lazy_preprocess True &