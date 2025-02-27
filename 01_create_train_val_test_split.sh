#!/bin/bash

#SBATCH --job-name=partition-data
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

hm="/gpfs/data/bbj-lab/users/$(whoami)"
cd "${hm}/clif-tokenizer" || exit
source ~/.bashrc
source venv/bin/activate
python3 01_create_train_val_test_split.py \
     --data_dir_in "${hm}/CLIF-MIMIC/output/rclif-2.1/" \
     --data_dir_out "${hm}/clif-data/TEST" \
     --train_frac 0.7 \
     --val_frac 0.1
