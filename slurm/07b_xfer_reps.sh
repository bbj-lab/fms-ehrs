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
    --data_version Y21_unfused_first_24h \
    --model_loc "${hm}/mdls-archive/gemma-5687290-Y21_unfused" \
    --classifier lr_pca \
    --k 25 \
    --save_preds

source postscript.sh
