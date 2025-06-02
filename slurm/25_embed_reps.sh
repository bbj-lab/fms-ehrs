#!/bin/bash

#SBATCH --job-name=cluster-reps
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=100GB
#SBATCH --time=1-00:00:00

source preamble.sh

models=(
    llama1b-original-59946215-hp-QC_noX
    llama1b-original-59946344-hp-QC_noX_sigmas
)

for m in "${models[@]}"; do
    python3 ../src/scripts/embed_reps.py \
        --data_dir_orig "${hm}/clif-data" \
        --data_dir_new "${hm}/clif-data-ucmc" \
        --data_version "${m##*-}_first_24h" \
        --model_loc "${hm}/clif-mdls-archive/$m" \
        --mapper pacmap
done
