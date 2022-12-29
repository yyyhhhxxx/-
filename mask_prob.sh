#!/bin/bash
report_to=wandb
n=2
output_dir=./pretrain/ai_unsup${n}/
python posttraining.py \
    --dataset_name ai_unsup \
    --do_train \
    --use_my_tokenizer False \
    --use_my_mask True \
    --per_device_train_batch_size 128 \
    --max_seq_length 64 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --save_strategy epoch \
    --output_dir ${output_dir} \
    --report_to ${report_to}