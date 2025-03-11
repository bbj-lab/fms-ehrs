#!/bin/bash

#SBATCH --job-name=all-states
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=sxmq
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

source ~/.bashrc
source venv/bin/activate
export hm=/gpfs/data/bbj-lab/users/burkh4rt
torchrun --nproc_per_node=8 13_extract_all_hidden_states.py \
    --data_dir ${hm}/clif-data \
    --data_version day_stays_qc_first_24h \
    --model_loc ${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630 \
    --small_batch_sz $((2 ** 4)) \
    --big_batch_sz $((2 ** 10))
