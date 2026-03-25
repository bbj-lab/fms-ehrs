#!/bin/bash

#SBATCH --job-name=extract-outcomes
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00
#SBATCH --array=0-1

source preamble.sh
export data_version=W++

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        data_dir="${hm}/data-mimic"
        ;;
    1)
        data_dir="${hm}/data-ucmc"
        ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

python3 ../fms_ehrs/scripts/extract_outcomes.py \
    --data_dir "$data_dir" \
    --ref_version "${data_version:-QC_noX}" \
    --data_version "${data_version:-QC_noX}_first_24h"
