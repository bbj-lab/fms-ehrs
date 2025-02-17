#!/bin/bash

#SBATCH --job-name=train-clif-mamba
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source ~/.bashrc
source venv/bin/activate
python3 03_train_mamba_no_truncation.py
