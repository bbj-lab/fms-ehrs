#!/bin/bash

#SBATCH --job-name=trfd-sft-cf
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=4:00:00
##SBATCH --depend=afterok:60281166

source preamble.sh

outcomes=(same_admission_death long_length_of_stay icu_admission imv_event)
models=(
    mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death-60275657_0-hp
    mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death-60275657_1-hp
    mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death-60275657_2-hp
    mdl-llama1b-57928921-run1-58134628-clsfr-long_length_of_stay-60275657_3-hp
    mdl-llama1b-57928921-run1-58134628-clsfr-long_length_of_stay-60275657_4-hp
    mdl-llama1b-57928921-run1-58134628-clsfr-long_length_of_stay-60275657_5-hp
    mdl-llama1b-57928921-run1-58165534-clsfr-icu_admission-60275657_6-hp
    mdl-llama1b-57928921-run1-58165534-clsfr-icu_admission-60275657_7-hp
    mdl-llama1b-57928921-run1-58165534-clsfr-icu_admission-60275657_8-hp
    mdl-llama1b-57928921-run1-58165531-clsfr-imv_event-60275657_9-hp
    mdl-llama1b-57928921-run1-58165531-clsfr-imv_event-60275657_10-hp
    mdl-llama1b-57928921-run1-58165531-clsfr-imv_event-60275657_11-hp
)

for i in "${!models[@]}"; do
    python3 ../fms_ehrs/scripts/process_sft_preds.py \
        --data_dir_orig "${hm}/clif-data" \
        --data_dir_new "${hm}/clif-data-ucmc" \
        --data_version QC_day_stays_first_24h \
        --model_sft_loc "${hm}/clif-mdls-archive/${models[$i]}" \
        --model_outlier_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
        --outcome "${outcomes[$((i / 3))]}" \
        --only_new True
done
