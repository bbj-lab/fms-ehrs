#!/bin/bash

#SBATCH --job-name=extract-states
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7

source preamble.sh

div=2
quo=$((SLURM_ARRAY_TASK_ID / div))
rem=$((SLURM_ARRAY_TASK_ID % div))

data_dirs=(
    "${hm}/clif-data"
    "${hm}/clif-data-ucmc"
)

if [ -z "${versions}" ]; then
    versions=(
        icu24h
        icu24h_top5-921
        icu24h_bot5-921
        icu24h_rnd5-921
    )
fi

torchrun --nproc_per_node=4 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    ../src/scripts/extract_hidden_states.py \
    --data_dir "${data_dirs[$rem]}" \
    --data_version "${versions[$quo]}_first_24h" \
    --model_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
    --batch_sz $((2 ** 5))
