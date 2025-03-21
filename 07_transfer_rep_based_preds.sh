#!/bin/bash

#SBATCH --job-name=transfer-rep-preds
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=2:00:00

source preamble.sh

python3 "${name}.py" \
    --data_dir_orig "${hm}/clif-data" \
    --data_dir_new "/scratch/$(whoami)/clif-data" \
    --data_version day_stays_qc_first_24h \
    --model_loc "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630"
