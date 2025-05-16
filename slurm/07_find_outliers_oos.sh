#!/bin/bash

#SBATCH --job-name=outliers-oos
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=10GB
#SBATCH --time=1:00:00
##SBATCH --dependency=afterok:59000111_[0-5]

source preamble.sh

models=(
    llama1b-original-59946215-hp-QC_noX
    llama1b-original-59946344-hp-QC_noX_sigmas
)

for m in "${models[@]}"; do
    python3 ../src/scripts/find_outliers_oos.py \
        --data_dir_orig "${hm}/clif-data" \
        --data_dir_new "${hm}/clif-data-ucmc" \
        --data_version "${m##*-}_first_24h" \
        --model_loc "${hm}/clif-mdls-archive/$m" \
        --out_dir "${hm}"
done
