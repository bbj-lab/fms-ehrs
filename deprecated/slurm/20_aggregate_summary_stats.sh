#!/bin/bash

#SBATCH --job-name=tbl-summary
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/aggregate_summary_stats.py \
    --data_dir "${hm}/data-mimic" \
    --data_version QC_day_stays_first_24h \
    --raw_version raw \
    --model_outlier_loc "${hm}/mdls-archive/llama1b-57928921-run1"

python3 ../fms_ehrs/scripts/aggregate_summary_stats.py \
    --data_dir "${hm}/data-ucmc" \
    --data_version QC_day_stays_first_24h \
    --raw_version QC \
    --model_outlier_loc "${hm}/mdls-archive/llama1b-57928921-run1"
