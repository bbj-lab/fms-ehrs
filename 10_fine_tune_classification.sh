#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=sxmq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00

source preamble.sh

torchrun --nproc_per_node=8 \
    "${name}.py" \
    --model_dir "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630" \
    --data_dir "${hm}/clif-data/day_stays_qc_first_24h-tokenized" \
    --out_dir "${hm}/clif-mdls" \
    --n_epochs 5 \
    --learning_rate 0.00004 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2
