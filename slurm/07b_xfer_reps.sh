#!/bin/bash

#SBATCH --job-name=xfer-rep-preds
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=10GB
#SBATCH --time=1:00:00

source preamble.sh

echo "Generating representation-based predictions..."

outcomes=(
    "same_admission_death"
    "long_length_of_stay"
    "ama_discharge"
    "hospice_discharge"
)

for classifier in light_gbm logistic_regression logistic_regression_cv; do
    python3 ../fms_ehrs/scripts/transfer_rep_based_preds.py \
        --data_dir_orig "${hm}/data-mimic" \
        --data_dir_new "${hm}/data-ucmc" \
        --data_version Y21_icu24_first_24h \
        --model_loc "${hm}/mdls-archive/gemma-5635921-Y21" \
        --classifier "$classifier" \
        --outcomes "${outcomes[@]}" \
        --save_preds
done

for classifier in light_gbm logistic_regression logistic_regression_cv; do
    python3 ../fms_ehrs/scripts/transfer_rep_based_preds.py \
        --data_dir_orig "${hm}/data-mimic" \
        --data_dir_new "${hm}/data-ucmc" \
        --data_version Y21_unfused_icu24_first_24h \
        --model_loc "${hm}/mdls-archive/gemma-5687290-Y21_unfused" \
        --classifier "$classifier" \
        --outcomes "${outcomes[@]}" \
        --save_preds
done

source postscript.sh
