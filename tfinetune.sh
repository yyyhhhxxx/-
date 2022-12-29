#!/bin/bash
report_to=wandb

seed=(42 111 222 333 444 555 666 777)
postmodel=YOUR_MODEL_NAME
model=YOUR_MODEL_NAME_OR_PATH

for dataset_name in 'acl_sup' 'sci_sup'
do
    for ((i=1;i<=5;i++))
    do
        task_name=${postmodel}_${dataset_name}
        output_dir=./fine/${task_name}/${i}/
        python finetuning.py \
          --model_name_or_path ${model} \
          --task_name ${task_name} \
          --dataset_name ${dataset_name} \
          --seed ${seed[${i}]} \
          --do_train \
          --do_eval \
          --do_predict \
          --use_my_tokenizer True \
          --logging_strategy epoch \
          --evaluation_strategy epoch \
          --save_strategy epoch \
          --max_seq_length 164 \
          --per_device_train_batch_size 20 \
          --per_device_eval_batch_size 20 \
          --learning_rate 5e-5 \
          --weight_decay 1e-2 \
          --num_train_epochs 20 \
          --output_dir ${output_dir} \
          --report_to ${report_to}
    done
done
