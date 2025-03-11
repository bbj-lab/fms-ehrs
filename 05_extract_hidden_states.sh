#!/bin/bash

#SBATCH --job-name=extract-states
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00

hm="/gpfs/data/bbj-lab/users/$(whoami)"
cd "${hm}/clif-tokenizer" || exit
source ~/.bashrc
source venv/bin/activate
torchrun --nproc_per_node=3 05_extract_hidden_states.py \
    --data_dir "${hm}/clif-data" \
    --data_version day_stays_qc_first_24h \
    --model_loc "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630" \
    --batch_sz $((2 ** 5))
