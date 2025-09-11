#!/bin/bash

#SBATCH --job-name=partition-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/create_splits_w_config.py \
    --data_dir_in "${hm}/data-raw/ucmc-2.1.0" \
    --data_dir_out "${hm}/development-sample-21" \
    --data_version_out raw \
    --config_loc "${hm}/fms-ehrs-reps/fms_ehrs/config/config-20.yaml" \
    --development_sample \
    --dev_frac 0.01
