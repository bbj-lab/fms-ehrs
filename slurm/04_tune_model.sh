#!/bin/bash

#SBATCH --job-name=tune-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00

source preamble.sh

[ -z "${data_version}" ] && export data_version=W++

echo "Training an FM on MIMIC data..."
torchrun --nproc_per_node=8 \
    ../fms_ehrs/scripts/tune_model.py \
    --n_epochs 10 \
    --n_trials 3 \
    --data_dir "${hm}/clif-data" \
    --data_version "${data_version}" \
    --model_dir "${hm}/clif-mdls" \
    --model_version "llama-med" \
    --model_name "meta-llama/Llama-3.2-1B" \
    --wandb_project "${data_version}" \
    --hidden_size 1024 \
    --intermediate_size 2048 \
    --num_hidden_layers 8 \
    --num_attention_heads 8

# this leaves tuned models at ${model_dir}/${model_version}-%j-hp-${data_version}
# typically we copy the ones we like to ${hm}/clif-mdls-archive
