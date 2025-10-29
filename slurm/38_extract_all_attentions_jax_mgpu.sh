#!/bin/bash

#SBATCH --job-name=xtract-attns-multi
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2

source preamble.sh

splits=("train" "val" "test")
metrics=(
    "h2o-mean"
    "h2o-mean_log"
    "h2o-va-mean"
    "h2o-va-mean_log"
    "scissorhands-10"
    "scissorhands-20"
    "scissorhands-va-10"
    "scissorhands-va-20"
    "h20-normed-mean"
    "h20-normed-mean_log"
)

torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    ../fms_ehrs/scripts/extract_all_attentions_jax.py \
    --data_dir "../../data-mimic" \
    --data_version "W++" \
    --model_loc "${hm}/mdls-archive/llama-med-60358922_1-hp-W++" \
    --batch_size 64 \
    --metrics "${metrics[@]}"
