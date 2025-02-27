#!/bin/bash

#SBATCH --job-name=examine-mdl
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

source ~/.bashrc
source venv/bin/activate
python3 04_examine_model.py \
        --projector_type PCA \
