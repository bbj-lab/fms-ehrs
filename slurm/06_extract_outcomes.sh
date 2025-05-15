#!/bin/bash

#SBATCH --job-name=extract-outcomes
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00
#SBATCH --array=0-1

source preamble.sh

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        data_dir="${hm}/clif-data"
        ;;
    1)
        data_dir="${hm}/clif-data-ucmc"
        ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

for data_version in QC_noX QC_noX_sigmas; do
    python3 ../src/scripts/extract_outcomes.py \
        --data_dir "$data_dir" \
        --ref_version "${data_version:-QC_noX}" \
        --data_version "${data_version:-QC_noX}_first_24h"
done
