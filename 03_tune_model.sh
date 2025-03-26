#!/bin/bash

#SBATCH --job-name=tune-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=sxmq
#SBATCH --gres=gpu:8
#SBATCH --time=10-00:00:00

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
    --wandb_project mimic-llama
