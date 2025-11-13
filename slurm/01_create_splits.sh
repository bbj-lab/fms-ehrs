#!/bin/bash

#SBATCH --job-name=partition-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=5GB
#SBATCH --time=1:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/partition_w_config.py \
    --data_dir_in "${hm}/data-raw/mimic-2.1.0/" \
    --data_dir_out "${hm}/data-mimic/" \
    --data_version_out W21 \
    --train_frac 0.7 \
    --val_frac 0.1 \
    --config_loc "../fms_ehrs/config/config-21.yaml"

python3 ../fms_ehrs/scripts/partition_w_config.py \
    --data_dir_in "${hm}/data-raw/ucmc-2.1.0/" \
    --data_dir_out "${hm}/data-ucmc" \
    --data_version_out W21 \
    --train_frac 0.05 \
    --val_frac 0.05 \
    --config_loc "../fms_ehrs/config/config-21.yaml"

source postscript.sh
