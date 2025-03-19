#!/bin/bash

#SBATCH --job-name=rep-based-preds
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=3:00:00

source preamble.sh

# for Mimic
python3 "${name}.py" \
    --data_dir "${hm}/clif-data" \
    --data_version day_stays_qc_first_24h \
    --model_loc "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630" \
    --fast false

# for UChicago
#python3 "${name}.py" \
#    --data_dir_orig "/scratch/$(whoami)/clif-data" \
#    --data_version day_stays_qc_first_24h \
#    --model_loc "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630" \
#    --fast false
