#!/bin/bash

#SBATCH --job-name=tune-mdl
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=sxmq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00

source ~/.bashrc
source venv/bin/activate
torchrun --nproc_per_node=8 03_tune_model_with_packing.py \
        --n_epochs 3 \
        --data_version day_stays_qc \
        --model_version llama1b \
        --model_name "meta-llama/Llama-3.2-1B"
