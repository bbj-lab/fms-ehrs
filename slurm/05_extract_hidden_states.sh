#!/bin/bash

#SBATCH --job-name=x-all-layers
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

ni=2 nj=5
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
    "llama1b-57928921-run1"
    "mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death"
    "mdl-llama1b-57928921-run1-58134628-clsfr-long_length_of_stay"
    "mdl-llama1b-57928921-run1-58165534-clsfr-icu_admission"
    "mdl-llama1b-57928921-run1-58165531-clsfr-imv_event"
)

torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    ../fms_ehrs/scripts/extract_hidden_states.py \
    --data_dir "${data_dirs[$i]}" \
    --data_version QC_day_stays_first_24h \
    --model_loc "${hm}/mdls-archive/${models[$j]}" \
    --batch_sz $((2 ** 3)) \
    --all_layers True
