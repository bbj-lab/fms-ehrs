#!/bin/bash

#SBATCH --job-name=partition-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/create_train_val_test_split.py \
    --data_dir_in "${hm}/CLIF-MIMIC/output/rclif-2.1-edit/" \
    --data_dir_out "${hm}/data-mimic/" \
    --data_version_out QC \
    --train_frac 0.7 \
    --val_frac 0.1

python3 ../fms_ehrs/scripts/create_train_val_test_split.py \
    --data_dir_in "${hm}/CLIF-2.0.0" \
    --data_dir_out "${hm}/data-ucmc" \
    --data_version_out QC \
    --train_frac 0.05 \
    --val_frac 0.05 \
    --valid_admission_window "('2020-03-01','2022-03-01')"
