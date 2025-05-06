#!/bin/bash

#SBATCH --job-name=tune-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:59626732
#SBATCH --array=0-3

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

names=(original med small smol)
hidden_sizes=(2048 1024 512 256)
intermediate_sizes=(8192 2048 1024 512)

res=$(
    torchrun --nproc_per_node=8 \
        ../src/scripts/tune_model.py \
        --n_epochs 10 \
        --n_trials 5 \
        --data_dir "${hm}/clif-data" \
        --data_version QC_no10 \
        --collation packed \
        --model_dir "${hm}/clif-mdls" \
        --model_version "llama1b-${names[$SLURM_ARRAY_TASK_ID]}" \
        --model_name "meta-llama/Llama-3.2-1B" \
        --wandb_project no10 \
        --hidden_size "${hidden_sizes[$SLURM_ARRAY_TASK_ID]}" \
        --intermediate_size "${intermediate_sizes[$SLURM_ARRAY_TASK_ID]}" \
        --num_hidden_layers $((SLURM_ARRAY_TASK_ID == 0 ? 2 ** 4 : 2 ** 3)) \
        --num_attention_heads $((SLURM_ARRAY_TASK_ID == 0 ? 2 ** 5 : 2 ** 3))
)

echo "$res"
