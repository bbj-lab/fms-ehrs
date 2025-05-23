#!/bin/bash

#SBATCH --job-name=cf-mdls
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00
##SBATCH --dependency=afterok:59000155_[0-6]
#SBATCH --array=0-1

source preamble.sh

case "${SLURM_ARRAY_TASK_ID}" in
    0) data_dir="${hm}/clif-data" ;;
    1) data_dir="${hm}/clif-data-ucmc" ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

models=(
    llama1b-original-59772926-hp
)

python3 ../src/scripts/aggregate_model_preds.py \
    --data_dir "${data_dir}" \
    --out_dir "${hm}/figs" \
    --data_version "${data_version:-QC_noX}_first_24h" \
    --models "${models[@]/#/${hm}/clif-mdls-archive/}"
