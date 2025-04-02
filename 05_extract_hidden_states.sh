#!/bin/bash

#SBATCH --job-name=extract-states
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=sxmq
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --array=1

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) data_dir="${hm}/clif-data" ;;
    1) data_dir="${hm}/clif-data-ucmc" ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

# "${hm}/clif-mdls-archive/mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death"
# "${hm}/clif-mdls-archive/mdl-llama1b-57928921-run1-58134628-clsfr-long_length_of_stay"
# "${hm}/clif-mdls-archive/mdl-llama1b-57928921-run1-58165531-clsfr-imv_event"
# "${hm}/clif-mdls-archive/mdl-llama1b-57928921-run1-58165534-clsfr-icu_admission"

torchrun --nproc_per_node=4 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    "${name}.py" \
    --data_dir "$data_dir" \
    --data_version QC_day_stays_first_24h \
    --model_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
    --batch_sz $((2 ** 5))
