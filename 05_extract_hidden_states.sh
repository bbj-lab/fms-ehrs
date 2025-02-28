#!/bin/bash

#SBATCH --job-name=extract-states
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00

source ~/.bashrc
source venv/bin/activate
export hm=/gpfs/data/bbj-lab/users/burkh4rt
torchrun --nproc_per_node=3 05_extract_hidden_states.py \
    --data_dir ${hm}/clif-data \
    --data_version day_stays_qc_first_24h \
    --model_loc ${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630 \
    --batch_sz $((2 ** 5))
