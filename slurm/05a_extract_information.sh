#!/bin/bash

#SBATCH --job-name=extract-info
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
splits=(train val test)

python3 ../fms_ehrs/scripts/extract_information.py \
    --data_dir "${data_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --data_version V21 \
    --model_loc "${hm}/mdls-archive/llama-med-4476655-hp-V21" \
    --batch_sz $((2 ** 5)) \
    --splits "${splits[@]}"

source postscript.sh
