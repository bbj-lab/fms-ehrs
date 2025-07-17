#!/bin/bash

#SBATCH --job-name=partition-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh

echo "Partitioning MIMIC data..."
python3 ../fms_ehrs/scripts/create_train_val_test_split.py \
    --data_dir_in "${hm}/CLIF-MIMICv0.1.0/output/rclif-2.0/" \
    --data_dir_out "${hm}/data-mimic/" \
    --data_version_out raw \
    --train_frac 0.7 \
    --val_frac 0.1

echo "Partitioning UCMC data..."
python3 ../fms_ehrs/scripts/create_train_val_test_split.py \
    --data_dir_in "/scratch/$(whoami)/CLIF-2.0.0" \
    --data_dir_out "${hm}/data-ucmc" \
    --data_version_out raw \
    --train_frac 0.05 \
    --val_frac 0.05 \
    --valid_admission_window "('2020-03-01','2022-03-01')"
