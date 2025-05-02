#!/bin/bash

#SBATCH --job-name=proc-log-probs
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

for m in "llama-med-58788824" "llama1b-57928921-run1"; do
    python3 ../src/scripts/process_log_probs.py \
        --data_dir_orig "${hm}/clif-data" \
        --data_dir_new "${hm}/clif-data-ucmc" \
        --data_version QC_day_stays_first_24h \
        --model_loc "${hm}/clif-mdls-archive/$m" \
        --out_dir "${hm}/figs"
done
