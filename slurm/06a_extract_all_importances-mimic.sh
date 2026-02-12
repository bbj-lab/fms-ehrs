#!/bin/bash

#SBATCH --job-name=xtract-imps
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-12

source preamble.sh

splits=("test")
metrics=(
    "h2o-mean"
    "h2o-va-mean"
    "h2o-normed-mean"
)

python3 ../fms_ehrs/scripts/extract_all_importances.py \
    --data_dir "${hm}/data-mimic" \
    --data_version Y21_icu24_first_24h \
    --model_loc "${hm}/mdls-archive/gemma-5635921-Y21" \
    --batch_size 32 \
    --metrics "${metrics[@]}" \
    --splits "${splits[@]}" \
    --batch_num_start $((1000 * SLURM_ARRAY_TASK_ID)) \
    --batch_num_end $((1000 * (SLURM_ARRAY_TASK_ID + 1))) \
    --use_jax

source postscript.sh
