#!/bin/bash

#SBATCH --job-name=xtract-all
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpudev
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-1%1

source preamble.sh

data_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
)

python3 ../fms_ehrs/scripts/extract_hidden_states.py \
    --data_dir "${data_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --data_version Y21_unfused_first_24h \
    --model_loc "${hm}/mdls-archive/gemma-5687290-Y21_unfused" \
    --batch_sz $((2 ** 5))

source postscript.sh
