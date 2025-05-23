#!/bin/bash

#SBATCH --job-name=outliers-oos
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=10GB
#SBATCH --time=1:00:00

source preamble.sh

if [ -z "${versions}" ]; then
    versions=(
        icu24h
        icu24h_top5-921
        icu24h_bot5-921
        icu24h_rnd5-921
    )
fi

for v in "${versions[@]}"; do
    python3 ../src/scripts/find_outliers_oos.py \
        --data_dir_orig "${hm}/clif-data" \
        --data_dir_new "${hm}/clif-data-ucmc" \
        --data_version "${v}_first_24h" \
        --model_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
        --out_dir "${hm}"
done
