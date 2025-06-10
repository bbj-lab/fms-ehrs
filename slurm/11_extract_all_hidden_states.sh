#!/bin/bash

#SBATCH --job-name=all-states
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=sxmq
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --array=0-1

source preamble.sh

case "${SLURM_ARRAY_TASK_ID}" in
    0) data_dir="${hm}/clif-data" ;;
    1) data_dir="${hm}/clif-data-ucmc" ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

torchrun --nproc_per_node=8 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    ../fms_ehrs/scripts/extract_all_hidden_states.py \
    --data_dir "$data_dir" \
    --data_version QC_day_stays_first_24h \
    --model_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
    --small_batch_sz $((2 ** 4)) \
    --big_batch_sz $((2 ** 12))
