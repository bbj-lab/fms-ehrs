#!/bin/bash

#SBATCH --job-name=cf-mdls
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=3:00:00
##SBATCH --dependency=afterok:58843812_[0-6]
#SBATCH --array=0-1

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) data_dir="${hm}/clif-data" ;;
    1) data_dir="${hm}/clif-data-ucmc" ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" ;;
esac

models=(
    "${hm}/clif-mdls-archive/llama-orig-58789721"
    "${hm}/clif-mdls-archive/llama-large-58788825"
    "${hm}/clif-mdls-archive/llama-med-58788824"
    "${hm}/clif-mdls-archive/llama-small-58741567"
    "${hm}/clif-mdls-archive/llama-smol-58761427"
    "${hm}/clif-mdls-archive/llama-tiny-58761428"
    "${hm}/clif-mdls-archive/llama-teensy-58741565"
)

python3 ../src/scripts/aggregate_model_preds.py \
    --data_dir "${data_dir}" \
    --out_dir "${hm}/figs" \
    --data_version QC_day_stays_first_24h \
    --models "${models[@]}"
