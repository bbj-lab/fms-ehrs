#!/bin/bash

#SBATCH --job-name=extract-outcomes
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00

source preamble.sh

for data_dir in "${hm}/data-mimic" "${hm}/data-ucmc"; do
    python3 ../fms_ehrs/scripts/filter_dataset.py \
        --data_dir "$data_dir" \
        --from_version "Y21_first_24h" \
        --to_version "Y21_icu24_first_24h" \
        --filter_col "icu_admission_24h"
done

for data_dir in "${hm}/data-mimic" "${hm}/data-ucmc"; do
    python3 ../fms_ehrs/scripts/filter_dataset.py \
        --data_dir "$data_dir" \
        --from_version "Y21_unfused_first_24h" \
        --to_version "Y21_unfused_icu24_first_24h" \
        --filter_col "icu_admission_24h"
done

source postscript.sh
