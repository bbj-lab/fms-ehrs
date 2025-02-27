#!/bin/bash

#SBATCH --job-name=examine-mdl
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

source ~/.bashrc
source venv/bin/activate
export hm=/gpfs/data/bbj-lab/users/burkh4rt
python3 04_examine_model.py \
        --projector_type PCA \
        --train_dir ${hm}/clif-data/day_stays_qc_first_24h-tokenized/train \
        --model_loc ${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630 \
        --out_dir ${hm}
