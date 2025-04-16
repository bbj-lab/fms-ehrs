#!/bin/bash

#SBATCH --job-name=tbl-summary
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

python3 ../src/scripts/aggregate_summary_stats.py \
    --data_dir "${hm}/clif-data" \
    --data_version QC_day_stays_first_24h \
    --raw_version raw \
    --model_outlier_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1"

python3 ../src/scripts/aggregate_summary_stats.py \
    --data_dir "${hm}/clif-data-ucmc" \
    --data_version QC_day_stays_first_24h \
    --raw_version QC \
    --model_outlier_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1"
