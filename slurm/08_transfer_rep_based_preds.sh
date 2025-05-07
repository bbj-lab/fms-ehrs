#!/bin/bash

#SBATCH --job-name=transfer-rep-preds
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00
#SBATCH --array=0-9
##SBATCH --dependency=afterok:59000124

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

models=(
    llama1b-original-59713433-hp-no10
    llama1b-med-59713434-hp-no10
    llama1b-small-59713435-hp-no10
    llama1b-smol-59713429-hp-no10
)

python3 ../src/scripts/transfer_rep_based_preds.py \
    --data_dir_orig "${hm}/clif-data" \
    --data_dir_new "${hm}/clif-data-ucmc" \
    --data_version QC_no10_first_24h \
    --model_loc "${hm}/clif-mdls-archive/${models[$SLURM_ARRAY_TASK_ID]}" \
    --classifier logistic_regression \
    --save_preds
