#!/bin/bash

#SBATCH --job-name=sft-cf
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=2:00:00

source preamble.sh

outcomes=(same_admission_death long_length_of_stay icu_admission imv_event)
models=(
    "mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death"
    "mdl-llama1b-57928921-run1-58134628-clsfr-long_length_of_stay"
    "mdl-llama1b-57928921-run1-58165534-clsfr-icu_admission"
    "mdl-llama1b-57928921-run1-58165531-clsfr-imv_event"
)

for i in {0..3}; do
    python3 "${name}.py" \
        --data_dir_orig "${hm}/clif-data" \
        --data_dir_new "${hm}/clif-data-ucmc" \
        --data_version QC_day_stays_first_24h \
        --model_sft_loc "${hm}/clif-mdls-archive/${models[$i]}" \
        --model_outlier_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
        --outcome "${outcomes[$i]}"
done
