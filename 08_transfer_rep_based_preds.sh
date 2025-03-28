#!/bin/bash

#SBATCH --job-name=transfer-rep-preds
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00

source preamble.sh

python3 "08_transfer_rep_based_preds.py" \
    --data_dir_orig "${hm}/clif-data" \
    --data_dir_new "/scratch/$(whoami)/clif-data" \
    --data_version QC_day_stays_first_24h \
    --model_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1"
