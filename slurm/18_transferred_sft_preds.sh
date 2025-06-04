#!/bin/bash

#SBATCH --job-name=tfr-preds
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11

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

python3 ../src/scripts/fine_tuned_predictions.py \
    --data_dir "${hm}/clif-data-ucmc" \
    --data_version QC_day_stays_first_24h \
    --model_loc "${hm}/clif-mdls-archive/${models[$SLURM_ARRAY_TASK_ID]}" \
    --outcome "${outcomes[$((SLURM_ARRAY_TASK_ID / 3))]}"
