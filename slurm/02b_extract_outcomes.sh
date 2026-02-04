#!/bin/bash

#SBATCH --job-name=extract-outcomes
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00

source preamble.sh

for data_dir in "${hm}/data-mimic" "${hm}/data-ucmc"; do
    python3 ../fms_ehrs/scripts/extract_outcomes.py \
        --data_dir "$data_dir" \
        --ref_version "Y21" \
        --data_version "Y21_first_24h"
done

source postscript.sh
