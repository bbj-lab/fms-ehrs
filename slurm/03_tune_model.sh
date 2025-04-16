#!/bin/bash

#SBATCH --job-name=tune-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00

source preamble.sh

torchrun --nproc_per_node=8 \
    ../src/scripts/tune_model.py \
    --n_epochs 5 \
    --data_dir "${hm}/clif-data" \
    --data_version QC_day_stays \
    --collation packed \
    --model_dir "${hm}/clif-mdls" \
    --model_version llama1b \
    --model_name "meta-llama/Llama-3.2-1B" \
    --wandb_project mimic-llama
