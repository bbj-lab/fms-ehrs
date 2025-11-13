#!/bin/bash

#SBATCH --job-name=partition-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=10GB
#SBATCH --time=1:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/partition_w_config.py \
    --data_dir_in "${hm}/data-raw/ucmc-2.1.0" \
    --data_dir_out "${hm}/development-sample-21" \
    --data_version_out raw-ucmc \
    --config_loc "${hm}/fms-ehrs-reps/fms_ehrs/config/config-21.yaml" \
    --development_sample \
    --dev_frac 0.01

python3 ../fms_ehrs/scripts/partition_w_config.py \
    --data_dir_in "${hm}/data-raw/mimic-2.1.0" \
    --data_dir_out "${hm}/development-sample-21" \
    --data_version_out raw-mimic \
    --config_loc "${hm}/fms-ehrs-reps/fms_ehrs/config/config-21.yaml" \
    --development_sample \
    --dev_frac 0.01

source postscript.sh
