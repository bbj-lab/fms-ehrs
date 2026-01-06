#!/bin/bash

#SBATCH --job-name=xtract-all
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --mem=160G

source preamble.sh

data_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
)
out_dirs=(
    "/scratch/burkh4rt/data-mimic"
    "/scratch/burkh4rt/data-ucmc"
)

python3 ../fms_ehrs/scripts/extract_all_hidden_states.py \
    --data_dir "${data_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --out_dir "${out_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --data_version V21 \
    --model_loc "${hm}/mdls-archive/llama-med-4476655-hp-V21" \
    --small_batch_sz $((2 ** 3)) \
    --big_batch_sz $((2 ** 10)) \
    --all_layers True

source postscript.sh
