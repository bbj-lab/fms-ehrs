#!/bin/bash

#SBATCH --job-name=examine-mdl
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

hm="/gpfs/data/bbj-lab/users/$(whoami)"
cd "${hm}/clif-tokenizer" || exit
source ~/.bashrc
source venv/bin/activate
python3 04_examine_model.py \
    --projector_type PCA \
    --data_dir "${hm}/clif-data/day_stays_qc_first_24h-tokenized/" \
    --model_loc "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630" \
    --out_dir "${hm}"
