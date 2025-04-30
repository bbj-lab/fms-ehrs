#!/bin/bash

#SBATCH --job-name=proc-log-probs
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

for data_dir in "${hm}/clif-data" "${hm}/clif-data-ucmc"; do
    python3 ../src/scripts/process_log_probs.py \
        --data_dir "$data_dir" \
        --data_version QC_day_stays_first_24h \
        --model_loc "${hm}/clif-mdls-archive/llama-med-58788824" \
        --out_dir "${hm}/figs"
done
