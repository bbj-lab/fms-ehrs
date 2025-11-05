#!/bin/bash

#SBATCH --job-name=get-attns
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

source preamble.sh

splits=("val")
metrics=(
    "h2o-mean"
    "h2o-mean_log"
    "h2o-va-mean"
    "h2o-va-mean_log"
    "scissorhands-10"
    "scissorhands-20"
    "scissorhands-va-10"
    "scissorhands-va-20"
    "rollout-mean"
    "rollout-mean_log"
    "h2o-normed-mean"
    "h2o-normed-mean_log"
)

python3 ../fms_ehrs/scripts/extract_all_attentions.py \
    --data_dir "../../data-ucmc" \
    --data_version "W++" \
    --model_loc "${hm}/mdls-archive/llama-med-60358922_1-hp-W++" \
    --batch_size 16 \
    --metrics "${metrics[@]}" \
    --splits "${splits[@]}" \
    --use_jax
