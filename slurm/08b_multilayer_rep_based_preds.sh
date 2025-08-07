#!/bin/bash

#SBATCH --job-name=multilayer-preds
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --array=0-4

source preamble.sh

models=(
    "llama1b-57928921-run1"
    "mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death"
    "mdl-llama1b-57928921-run1-58134628-clsfr-long_length_of_stay"
    "mdl-llama1b-57928921-run1-58165534-clsfr-icu_admission"
    "mdl-llama1b-57928921-run1-58165531-clsfr-imv_event"
)

for classifier in "light_gbm" "logistic_regression"; do
    python3 ../fms_ehrs/scripts/multilayer_rep_based_preds.py \
        --data_dir_orig "${hm}/data-mimic" \
        --data_dir_new "${hm}/data-ucmc" \
        --out_dir "${hm}/figs" \
        --data_version QC_day_stays_first_24h \
        --model_loc "${hm}/mdls-archive/${models[$SLURM_ARRAY_TASK_ID]}" \
        --classifier "${classifier}"
done
