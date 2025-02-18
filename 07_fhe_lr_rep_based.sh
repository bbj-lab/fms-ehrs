#!/bin/bash

#SBATCH --job-name=fhe-full-lr
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=sxmq
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00

source ~/.bashrc
conda activate concrete
python3 07_fhe_lr_rep_based.py
