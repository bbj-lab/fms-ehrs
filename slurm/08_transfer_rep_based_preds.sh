#!/bin/bash

#SBATCH --job-name=transfer-rep-preds
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00
#SBATCH --array=0-3

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

models=(
    llama-med-58788824
    llama-small-58741567
    llama-smol-58761427
    llama-teensy-58741565
    llama-tiny-58761428
)

python3 ../src/scripts/transfer_rep_based_preds.py \
    --data_dir_orig "${hm}/clif-data" \
    --data_dir_new "${hm}/clif-data-ucmc" \
    --data_version QC_day_stays_first_24h \
    --model_loc "${hm}/clif-mdls-archive/${models[$SLURM_ARRAY_TASK_ID]}" \
    --classifier logistic_regression
