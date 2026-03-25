#!/bin/bash

#SBATCH --job-name=add-lr-over-time
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=1TB
#SBATCH --time=3:00:00
#SBATCH --array=0-1

source preamble.sh

case "${SLURM_ARRAY_TASK_ID}" in
    0) data_dir="${hm}/data-mimic" ;;
    1) data_dir="${hm}/data-ucmc" ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

python3 ../fms_ehrs/scripts/lr_predictions_over_time.py \
    --data_dir_train "${hm}/data-mimic" \
    --data_dir_pred "$data_dir" \
    --data_version QC_day_stays_first_24h \
    --model_loc_base "${hm}/mdls-archive/llama1b-57928921-run1" \
    --model_loc_sft "${hm}/mdls-archive/mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death" \
    --big_batch_sz $((2 ** 12))
