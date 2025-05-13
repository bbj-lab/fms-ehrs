#!/bin/bash

#SBATCH --job-name=extract-outcomes
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00
#SBATCH --array=0-1

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        data_dir="${hm}/clif-data"
        ;;
    1)
        data_dir="${hm}/clif-data-ucmc"
        ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

python3 ../src/scripts/extract_outcomes.py \
    --data_dir "$data_dir" \
    --ref_version "${data_version:-QC_noX}" \
    --data_version "${data_version:-QC_noX}_first_24h"
