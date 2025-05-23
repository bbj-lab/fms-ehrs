#!/bin/bash

#SBATCH --job-name=transfer-rep-preds
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00
#SBATCH --array=0-3

source preamble.sh

if [ -z "${versions}" ]; then
    versions=(
        icu24h
        icu24h_top5-921
        icu24h_bot5-921
        icu24h_rnd5-921
    )
fi

python3 ../src/scripts/transfer_rep_based_preds.py \
    --data_dir_orig "${hm}/clif-data" \
    --data_dir_new "${hm}/clif-data-ucmc" \
    --data_version "${versions[$SLURM_ARRAY_TASK_ID]}_first_24h" \
    --model_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
    --classifier logistic_regression \
    --save_preds \
    --drop_icu_adm
