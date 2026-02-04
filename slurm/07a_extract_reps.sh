#!/bin/bash

#SBATCH --job-name=xtract-all
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpudev
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-5%1

source preamble.sh

ni=2 nj=3
i=$((SLURM_ARRAY_TASK_ID % ni)) j=$((SLURM_ARRAY_TASK_ID / ni))

if ((SLURM_ARRAY_TASK_COUNT != ni * nj)); then
    echo "Warning:"
    echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"
    echo "ni*nj=$((ni * nj))"
fi

data_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
)
out_dirs=(
    "/scratch/burkh4rt/data-mimic"
    "/scratch/burkh4rt/data-ucmc"
)
splits=(
    train
    val
    test
)

python3 ../fms_ehrs/scripts/extract_all_hidden_states.py \
    --data_dir "${data_dirs[$i]}" \
    --out_dir "${out_dirs[$i]}" \
    --data_version Y21 \
    --model_loc "${hm}/mdls-archive/gemma-5635921-Y21" \
    --small_batch_sz $((2 ** 3)) \
    --big_batch_sz $((2 ** 10)) \
    --all_layers \
    --splits "[\"${splits[$j]}\"]"

source postscript.sh
