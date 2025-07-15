#!/bin/bash

#SBATCH --job-name=extract-log-probs
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-1

source preamble.sh

model=llama-med-60358922_1-hp-W++
data_dirs=("${hm}/clif-data" "${hm}/clif-data-ucmc")
splits=(train val test)

torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    ../fms_ehrs/scripts/extract_log_probs.py \
    --data_dir "${data_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --data_version "${model##*-}" \
    --model_loc "${hm}/clif-mdls-archive/${model}" \
    --batch_sz $((2 ** 5)) \
    --splits "${splits[@]}"
