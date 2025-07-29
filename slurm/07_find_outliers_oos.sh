#!/bin/bash

#SBATCH --job-name=outliers-oos
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=10GB
#SBATCH --time=1:00:00

source preamble.sh

if [ -z "${versions}" ]; then
    versions=(
        W++
    )
fi

models=(
    llama-original-60358922_0-hp-W++
    llama-med-60358922_1-hp-W++
    llama-small-60358922_2-hp-W++
    llama-smol-60358922_3-hp-W++
)

for v in "${versions[@]}"; do
    for m in "${models[@]}"; do
        python3 ../fms_ehrs/scripts/find_outliers_oos.py \
            --data_dir_orig "${hm}/data-mimic" \
            --data_dir_new "${hm}/data-ucmc" \
            --data_version "${v}_first_24h" \
            --model_loc "${hm}/clif-mdls-archive/${m}" \
            --out_dir "${hm}"
    done
done
