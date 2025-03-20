#!/bin/bash

#SBATCH --job-name=train-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=sxmq
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

hm="/gpfs/data/bbj-lab/users/$(whoami)"
cd "${hm}/clif-tokenizer" || exit
source ~/.bashrc
source venv/bin/activate
torchrun --nproc_per_node=8 \
    03_train_model.py \
    --n_epochs 10 \
    --data_dir "${hm}/clif-data" \
    --data_version day_stays_qc \
    --collation packed \
    --model_dir "${hm}/clif-mdls" \
    --model_version mamba130m \
    --model_name "state-spaces/mamba-130m-hf" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 3 \
    --learning_rate 0.0002 \
    --hidden_size 768 \
    --n_layer 24 \
    --num_hidden_layers 24 \
    --state_size 16
