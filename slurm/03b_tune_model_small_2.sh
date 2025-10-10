#!/bin/bash

#SBATCH --job-name=small-2
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00

source preamble.sh

export data_version=W21

echo "Training an FM on MIMIC data..."
torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    ../fms_ehrs/scripts/tune_model.py \
    --n_epochs 10 \
    --n_trials 7 \
    --per_device_train_batch_size 8 \
    --data_dir "${hm}/data-mimic" \
    --data_version "${data_version}" \
    --model_dir "${hm}/mdls" \
    --model_version llama-small \
    --model_name "meta-llama/Llama-3.2-1B" \
    --wandb_project ${data_version} \
    --iterable_dataset False \
    --hidden_size 1024 \
    --intermediate_size 2048 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --attention_dropout 0.1

# this leaves tuned models at ${model_dir}/${model_version}-%j-hp-${data_version}
# typically we copy the ones we like to ${hm}/mdls-archive
