#!/bin/bash

#SBATCH --job-name=tune-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
##SBATCH --partition=sxmq
##SBATCH --reservation=sxmtest
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
##SBATCH --depend=afterok:3093746

source preamble.sh

export data_version=V21

echo "Training an FM on MIMIC data..."
torchrun --nproc_per_node=8 \
    --rdzv_backend c10d \
    --rdzv-id "${SLURM_ARRAY_TASK_ID:-0}" \
    --rdzv-endpoint=localhost:0 \
    ../fms_ehrs/scripts/tune_model.py \
    --n_epochs 40 \
    --n_trials 5 \
    --lr_min 2e-4 \
    --lr_max 3e-4 \
    --gr_acc_min 1 \
    --gr_acc_max 1 \
    --data_dir "${hm}/data-mimic" \
    --data_version "${data_version}" \
    --model_dir "${hm}/mdls" \
    --model_version llama-med \
    --model_name "meta-llama/Llama-3.2-1B" \
    --wandb_project ${data_version} \
    --hidden_size 1024 \
    --intermediate_size 2048 \
    --num_hidden_layers 8 \
    --num_attention_heads 8

# this leaves tuned models at ${model_dir}/${model_version}-%j-hp-${data_version}
# typically we copy the ones we like to ${hm}/mdls-archive
