#!/bin/bash

#SBATCH --job-name=xfer-rep-preds
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=10GB
#SBATCH --time=1:00:00

source preamble.sh

echo "Generating representation-based predictions..."
python3 ../fms_ehrs/scripts/transfer_rep_based_preds.py \
    --data_dir_orig "${hm}/data-mimic" \
    --data_dir_new "${hm}/data-ucmc" \
    --data_version Y21_first_24h \
    --model_loc "${hm}/mdls-archive/gemma-5635921-Y21" \
    --classifier logistic_regression_cv \
    --save_preds

source postscript.sh
