#!/bin/bash

#SBATCH --job-name=all-states
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --array=0-1

source preamble.sh

div=2
quo=$((SLURM_ARRAY_TASK_ID / div))
rem=$((SLURM_ARRAY_TASK_ID % div))

model=llama-med-60358922_1-hp-W++
data_dirs=("${hm}/clif-data" "${hm}/clif-data-ucmc")
out_dirs="${data_dirs[*]}"

torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    ../fms_ehrs/scripts/extract_all_hidden_states.py \
    --data_dir "${data_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --out_dir "${out_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --data_version "${model##*-}" \
    --model_loc "${hm}/clif-mdls-archive/${model}" \
    --small_batch_sz $((2 ** 4)) \
    --big_batch_sz $((2 ** 12)) \
    --test_only True
