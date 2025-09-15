#!/bin/bash

#SBATCH --job-name=tune-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00

source preamble.sh

echo "Training an FM on MIMIC data..."
torchrun --nproc_per_node=8 \
    ../fms_ehrs/scripts/tune_model.py \
    --n_epochs 10 \
    --n_trials 10 \
    --data_dir "${hm}/data-mimic" \
    --data_version "${data_version}" \
    --model_dir "${hm}/mdls" \
    --model_version W21 \
    --model_name "meta-llama/Llama-3.2-1B" \
    --wandb_project W21

# this leaves tuned models at ${model_dir}/${model_version}-%j-hp-${data_version}
# typically we copy the ones we like to ${hm}/mdls-archive
