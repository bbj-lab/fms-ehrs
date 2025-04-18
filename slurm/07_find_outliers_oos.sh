#!/bin/bash

#SBATCH --job-name=outliers-oos
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=25GB
#SBATCH --time=2:00:00
#SBATCH --dependency=afterok:58822890_0:58822890_2

source preamble.sh

models=(
    llama-orig-58789721
    llama-large-58788825
    llama-med-58788824
    llama-small-58741567
    llama-smol-58761427
    llama-tiny-58761428
    llama-teensy-58741565
)

for m in "${models[@]}"; do
    python3 ../src/scripts/find_outliers_oos.py \
        --data_dir_orig "${hm}/clif-data" \
        --data_dir_new "${hm}/clif-data-ucmc" \
        --data_version QC_day_stays_first_24h \
        --model_loc "${hm}/clif-mdls-archive/$m" \
        --out_dir "${hm}"
done
