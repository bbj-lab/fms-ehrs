#!/bin/bash

#SBATCH --job-name=tfr-preds
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

outcomes=(same_admission_death long_length_of_stay icu_admission imv_event)
models=(
    "mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death-58321088-run4"
    "mdl-llama1b-57928921-run1-58134628-clsfr-long_length_of_stay-58321089-run4"
    "mdl-llama1b-57928921-run1-58165534-clsfr-icu_admission-58321090-run2"
    "mdl-llama1b-57928921-run1-58165531-clsfr-imv_event-58321087-run0"
)

python3 10_fine_tuned_predictions.py \
    --data_dir "${hm}/clif-data-ucmc" \
    --data_version QC_day_stays_first_24h \
    --model_loc "${hm}/clif-mdls-archive/${models[$SLURM_ARRAY_TASK_ID]}" \
    --outcome "${outcomes[$SLURM_ARRAY_TASK_ID]}"
