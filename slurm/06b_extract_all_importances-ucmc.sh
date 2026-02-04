#!/bin/bash

#SBATCH --job-name=xtract-imps
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-108

source preamble.sh

splits=("test")
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

python3 ../fms_ehrs/scripts/extract_all_importances.py \
    --data_dir "${hm}/data-ucmc" \
    --data_version Y21 \
    --model_loc "${hm}/mdls-archive/gemma-5635921-Y21" \
    --batch_size 8 \
    --metrics "${metrics[@]}" \
    --splits "${splits[@]}" \
    --batch_num_start $((100 * SLURM_ARRAY_TASK_ID)) \
    --batch_num_end $((100 * (SLURM_ARRAY_TASK_ID + 1))) \
    --use_jax

source postscript.sh
