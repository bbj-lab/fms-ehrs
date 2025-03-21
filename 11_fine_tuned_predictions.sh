#!/bin/bash

#SBATCH --job-name=eval-ft-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-1

source preamble.sh

case "${SLURM_ARRAY_TASK_ID}" in
    0) data_dir="${hm}/clif-data/day_stays_qc_first_24h-tokenized" ;;
    1) data_dir="/scratch/burkh4rt/clif-data/day_stays_qc_first_24h-tokenized" ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

python3 "${name}.py" \
    --data_dir "${data_dir}" \
    --model_dir "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630-57723914-clsfr" \
    --outcome long_length_of_stay
