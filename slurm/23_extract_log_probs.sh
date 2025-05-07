#!/bin/bash

#SBATCH --job-name=extract-log-probs
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=sxmq
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --array=0-7

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

div=2
quo=$((SLURM_ARRAY_TASK_ID / div))
rem=$((SLURM_ARRAY_TASK_ID % div))

data_dirs=(
    "${hm}/clif-data"
    "${hm}/clif-data-ucmc"
)
models=(
    llama1b-original-59713433-hp-no10
    llama1b-med-59713434-hp-no10
    llama1b-small-59713435-hp-no10
    llama1b-smol-59713429-hp-no10
)

torchrun --nproc_per_node=4 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    ../src/scripts/extract_log_probs.py \
    --data_dir "${data_dirs[$rem]}" \
    --data_version QC_no10_first_24h \
    --model_loc "${hm}/clif-mdls-archive/${models[$quo]}" \
    --batch_sz $((2 ** 5))
