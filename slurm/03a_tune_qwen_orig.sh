#!/bin/bash

#SBATCH --job-name=qwen-orig
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
    --n_trials 5 \
    --per_device_train_batch_size 8 \
    --data_dir "${hm}/data-mimic" \
    --data_version "${data_version}" \
    --model_dir "${hm}/mdls" \
    --model_version qwen-orig \
    --model_name "Qwen/Qwen2-0.5B" \
    --wandb_project ${data_version} \
    --iterable_dataset False \
    --lr_min 5e-5 \
    --lr_max 1e-4

# --max_grad_norm 0.1 #
# --attention_dropout 0.1 \
#    --gr_acc_min 1 \
#    --gr_acc_max 1

# this leaves tuned models at ${model_dir}/${model_version}-%j-hp-${data_version}
# typically we copy the ones we like to ${hm}/mdls-archive
