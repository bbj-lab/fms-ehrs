#!/bin/bash

#SBATCH --job-name=partition-data
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=tier3q
#SBATCH --mem=1TB
#SBATCH --time=1:00:00

source ~/.bashrc
source venv/bin/activate
python3 01_create_train_val_test_split.py \
     --version_name raw \
     --train_frac 0.7 \
     --val_frac 0.1
