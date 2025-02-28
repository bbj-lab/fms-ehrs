#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=sxmq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00

source ~/.bashrc
source venv/bin/activate
export hm=/gpfs/data/bbj-lab/users/burkh4rt
torchrun --nproc_per_node=8 \
    10_fine_tune_classification.py \
    --model_dir ${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630 \
    --data_dir ${hm}/clif-data/day_stays_qc_first_24h-tokenized \
    --out_dir ${hm}/clif-mdls \
    --model_version llama1b-sft \
    --n_epochs 10 \
    --learning_rate 0.00002 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2
