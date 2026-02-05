#!/bin/bash

#SBATCH --job-name=xtract-reps
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpudev
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3%1

source preamble.sh

ni=2 nj=2
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
versions=(
    Y21_unfused_first_24h
    Y21_first_24h
)
models=(
    gemma-5687290-Y21_unfused
    gemma-5635921-Y21
)

python3 ../fms_ehrs/scripts/extract_hidden_states.py \
    --data_dir "${data_dirs[$i]}" \
    --data_version "${versions[$j]}" \
    --model_loc "${hm}/mdls-archive/${models[$j]}" \
    --batch_sz $((2 ** 5))

source postscript.sh
