#!/bin/bash

#SBATCH --job-name=extract-outcomes
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00
#SBATCH --array=0-1

source preamble.sh

data_dirs=("${hm}/data-mimic" "${hm}/data-ucmc")

echo "Extracting outcomes..."
python3 ../fms_ehrs/scripts/extract_outcomes.py \
    --data_dir "${data_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --ref_version W++ \
    --data_version W++_first_24h
