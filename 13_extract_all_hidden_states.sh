#!/bin/bash

#SBATCH --job-name=all-states
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --array=0-1

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) data_dir="${hm}/clif-data" ;;
    1) data_dir="/scratch/burkh4rt/clif-data" ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

torchrun --nproc_per_node=4 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    "${name}.py" \
    --data_dir "$data_dir" \
    --data_version QC_day_stays \
    --model_loc "${hm}/clif-mdls-archive/mdl-QC_day_stays-llama1b-57895023" \
    --small_batch_sz $((2 ** 4)) \
    --big_batch_sz $((2 ** 12))
