#!/bin/bash

#SBATCH --job-name=stats
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

model=llama-med-60358922_1-hp-W++
data_dirs=("${hm}/clif-data" "${hm}/clif-data-ucmc")

for d in "${data_dirs[@]}"; do
    python3 ../fms_ehrs/scripts/process_stats.py \
        --data_dir "$d" \
        --data_version "${model##*-}_first_24h" \
        --model_loc "${hm}/clif-mdls-archive/${model}"
done
