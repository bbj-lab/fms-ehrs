#!/bin/bash

#SBATCH --job-name=xtract-attns
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

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

python3 ../fms_ehrs/scripts/extract_all_attentions.py \
    --data_dir "../../data-mimic" \
    --data_version "W++" \
    --model_loc "${hm}/mdls-archive/llama-med-60358922_1-hp-W++" \
    --batch_size 32 \
    --metrics "${metrics[@]}"
