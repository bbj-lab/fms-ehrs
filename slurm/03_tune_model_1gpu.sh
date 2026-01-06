#!/bin/bash

#SBATCH --job-name=tune-mdl
#SBATCH --output=./output/%j-%x.stdout
##SBATCH --partition=sxmq
##SBATCH --reservation=sxmtest
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
##SBATCH --depend=afterok:3093746

source preamble.sh

export data_version=V21

echo "Training an FM on MIMIC data..."
python3 ../fms_ehrs/scripts/tune_model.py \
    --n_epochs 5 \
    --n_trials 10 \
    --lr_min 2e-4 \
    --lr_max 4e-4 \
    --gr_acc_min 1 \
    --gr_acc_max 4 \
    --per_device_train_batch_size 8 \
    --max_seq_length 4096 \
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
