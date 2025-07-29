#!/bin/bash

#SBATCH --job-name=extract-log-probs
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-1

source preamble.sh

div=2
quo=$((SLURM_ARRAY_TASK_ID / div))
rem=$((SLURM_ARRAY_TASK_ID % div))

data_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
)
models=(
    #    llama-original-60358922_0-hp-W++
    llama-med-60358922_1-hp-W++
    #    llama-small-60358922_2-hp-W++
    #    llama-smol-60358922_3-hp-W++
)
splits=(train val test)

torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    ../fms_ehrs/scripts/extract_log_probs.py \
    --data_dir "${data_dirs[$rem]}" \
    --data_version "${models[$quo]##*-}" \
    --model_loc "${hm}/clif-mdls-archive/${models[$quo]}" \
    --batch_sz $((2 ** 5)) \
    --splits "${splits[@]}"
