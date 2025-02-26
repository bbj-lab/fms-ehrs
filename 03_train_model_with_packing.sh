#!/bin/bash

#SBATCH --job-name=train-clif-mamba
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

source ~/.bashrc
source venv/bin/activate
torchrun --nproc_per_node=2 \
         --master_port=29501 \
      03_train_model_with_packing.py \
        --n_epochs 10 \
        --data_version day_stays_qc \
        --model_version llama1b \
        --model_name "meta-llama/Llama-3.2-1B" \
        --per_device_train_batch_size  4 \
        --per_device_eval_batch_size 4
