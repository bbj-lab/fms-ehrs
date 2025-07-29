#!/bin/bash

#SBATCH --job-name=tune-mdl
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3

source preamble.sh

[ -z "${data_version}" ] && export data_version=W++

names=(original med small smol)
hidden_sizes=(2048 1024 512 256)
intermediate_sizes=(8192 2048 1024 512)

torchrun --nproc_per_node=8 \
    ../fms_ehrs/scripts/tune_model.py \
    --n_epochs 10 \
    --n_trials 3 \
    --data_dir "${hm}/data-mimic" \
    --data_version "${data_version:-QC_noX}" \
    --model_dir "${hm}/clif-mdls" \
    --model_version "llama-${names[$SLURM_ARRAY_TASK_ID]}" \
    --model_name "meta-llama/Llama-3.2-1B" \
    --wandb_project "${data_version:-QC_noX}" \
    --jid "'${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'" \
    --hidden_size "${hidden_sizes[$SLURM_ARRAY_TASK_ID]}" \
    --intermediate_size "${intermediate_sizes[$SLURM_ARRAY_TASK_ID]}" \
    --num_hidden_layers $((SLURM_ARRAY_TASK_ID == 0 ? 2 ** 4 : 2 ** 3)) \
    --num_attention_heads $((SLURM_ARRAY_TASK_ID == 0 ? 2 ** 5 : 2 ** 3))

# this leaves tuned models at ${model_dir}/${model_version}-${jid}-hp-${data_version}
