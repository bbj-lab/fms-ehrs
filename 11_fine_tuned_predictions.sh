#!/bin/bash

#SBATCH --job-name=eval-ft-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) data_dir="${hm}/clif-data" ;;
    1) data_dir="/scratch/burkh4rt/clif-data" ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

echo "fine-tuned chicago preds..."
python3 "${name}.py" \
    --data_dir "$data_dir" \
    --data_version QC_day_stays_first_24h \
    --model_loc "${hm}/clif-mdls-archive/mdl-mdl-day_stays_qc-llama1b-57350630-57748149-clsfr-long_length_of_stay" \
    --outcome long_length_of_stay
