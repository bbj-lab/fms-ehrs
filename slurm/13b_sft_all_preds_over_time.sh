#!/bin/bash

#SBATCH --job-name=sft-all-preds-over-time
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

ni=2 nj=2
i=$((SLURM_ARRAY_TASK_ID % ni)) j=$((SLURM_ARRAY_TASK_ID / ni))

if ((SLURM_ARRAY_TASK_COUNT != ni * nj)); then
    echo "Warning:"
    echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"
    echo "ni*nj=$((ni * nj))"
fi

data_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
)
models=(
    "mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death"
    "mdl-llama1b-57928921-run1-58148405-clsfr-same_admission_death-urt"
)

python3 ../fms_ehrs/scripts/sft_predictions_over_time.py \
    --data_dir "${data_dirs[i]}" \
    --data_version QC_day_stays_first_24h \
    --sft_model_loc "${hm}/mdls-archive/${models[j]}"
