#!/bin/bash

#SBATCH --job-name=partition-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh

tables=(
    # clif_respiratory_support_processed
    # clif_sofa
    # clif_successful_extubation
    clif_medication_admin_continuous_converted
    clif_medication_admin_intermittent_converted
)

for tbl in "${tables[@]}"; do

    python3 ../fms_ehrs/scripts/partition_posthoc_w_config.py \
        --new_data_loc "${hm}/data-raw/mimic-2.1.0/${tbl}.parquet" \
        --data_dir "${hm}/data-mimic/" \
        --data_version W21 \
        --config_loc "../fms_ehrs/config/config-21.yaml"

    python3 ../fms_ehrs/scripts/partition_posthoc_w_config.py \
        --new_data_loc "${hm}/data-raw/ucmc-2.1.0/${tbl}.parquet" \
        --data_dir "${hm}/data-ucmc/" \
        --data_version W21 \
        --config_loc "../fms_ehrs/config/config-21.yaml"

    python3 ../fms_ehrs/scripts/partition_posthoc_w_config.py \
        --new_data_loc "${hm}/data-raw/mimic-2.1.0/${tbl}.parquet" \
        --data_dir "${hm}/development-sample-21/" \
        --data_version raw-mimic \
        --config_loc "../fms_ehrs/config/config-21.yaml" \
        --development_sample

    python3 ../fms_ehrs/scripts/partition_posthoc_w_config.py \
        --new_data_loc "${hm}/data-raw/ucmc-2.1.0/${tbl}.parquet" \
        --data_dir "${hm}/development-sample-21/" \
        --data_version raw-ucmc \
        --config_loc "../fms_ehrs/config/config-21.yaml" \
        --development_sample

done

source postscript.sh
