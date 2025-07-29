#!/bin/bash

#SBATCH --job-name=transfer-rep-preds
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=1:00:00
#SBATCH --array=0-3

source preamble.sh

models=(
    llama-original-60358922_0-hp-W++
    llama-med-60358922_1-hp-W++
    llama-small-60358922_2-hp-W++
    llama-smol-60358922_3-hp-W++
)

python3 ../fms_ehrs/scripts/transfer_rep_based_preds.py \
    --data_dir_orig "${hm}/data-mimic" \
    --data_dir_new "${hm}/data-ucmc" \
    --data_version "${models[$SLURM_ARRAY_TASK_ID]##*-}_first_24h" \
    --model_loc "${hm}/clif-mdls-archive/${models[$SLURM_ARRAY_TASK_ID]}" \
    --classifier logistic_regression \
    --save_preds
