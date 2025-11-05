#!/bin/bash

#SBATCH --job-name=get-sals
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

source preamble.sh

ni=2 nj=4
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
outcomes=(
    "same_admission_death"
    "long_length_of_stay"
    "icu_admission"
    "imv_event"
)
models=(
    llama-med-60358922_1-hp-W++-sft-mort
    llama-med-60358922_1-hp-W++-sft-llos
    llama-med-60358922_1-hp-W++-sft-icua
    llama-med-60358922_1-hp-W++-sft-imve
)

python3 ../fms_ehrs/scripts/extract_all_saliency.py \
    --data_dir "${data_dirs[$i]}" \
    --data_version "W++" \
    --model_loc "${hm}/mdls-archive/${models[$j]}" \
    --outcome "${outcomes[$j]}" \
    --batch_size 64
