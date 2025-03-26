#!/bin/bash

#SBATCH --job-name=train-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

source preamble.sh

torchrun --nproc_per_node=8 \
    "${name}.py" \
    --n_epochs 10 \
    --data_dir "${hm}/clif-data" \
    --data_version QC_day_stays \
    --collation packed \
    --model_dir "${hm}/clif-mdls" \
    --model_version llama1b \
    --model_name "meta-llama/Llama-3.2-1B" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 3 \
    --learning_rate 0.0002 \
    --wandb_project mimic-llama
