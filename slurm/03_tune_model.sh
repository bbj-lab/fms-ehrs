#!/bin/bash

#SBATCH --job-name=tn-gemma
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
##SBATCH --dependency=afterok:5560968

source preamble.sh

export data_version=Y21

echo "Training an FM on MIMIC data..."
python3 ../fms_ehrs/scripts/tune_model.py \
    --n_epochs 10 \
    --n_trials 25 \
    --lr_min 2e-4 \
    --lr_max 4e-4 \
    --gr_acc_min 1 \
    --gr_acc_max 4 \
    --per_device_train_batch_size 4 \
    --max_seq_length 8192 \
    --data_dir "${hm}/data-mimic" \
    --data_version ${data_version} \
    --model_dir "${hm}/mdls" \
    --model_version gemma \
    --model_name "google/gemma-3-270m" \
    --wandb_project ${data_version}

source postscript.sh

# this leaves tuned models at ${model_dir}/${model_version}-%j-hp-${data_version}
# typically we copy the ones we like to ${hm}/mdls-archive
