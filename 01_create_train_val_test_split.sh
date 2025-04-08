#!/bin/bash

#SBATCH --job-name=partition-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh
dataset=mimic

case "$dataset" in
    mimic)
        python3 "${name}.py" \
            --data_dir_in "${hm}/CLIF-MIMIC/output/rclif-2.1-edit/" \
            --data_dir_out "${hm}/clif-data" \
            --data_version_out QC \
            --train_frac 0.7 \
            --val_frac 0.1
        ;;
    uchicago)
        python3 "${name}.py" \
            --data_dir_in "/scratch/$(whoami)/CLIF-2.0.0" \
            --data_dir_out "${hm}/clif-data-ucmc" \
            --data_version_out QC \
            --train_frac 0.05 \
            --val_frac 0.05 \
            --valid_admission_window "('2020-03-01','2022-03-01')"
        ;;
    *) echo "Invalid dataset spec: $dataset" ;;
esac
