#!/bin/bash

#SBATCH --job-name=cf-perf-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

data_dirs=(
    "${hm}/clif-data"
    "${hm}/clif-data-ucmc"
)
versions=(
    icu24h_first_24h
    icu24h_top5-921_first_24h
    icu24h_bot5-921_first_24h
    icu24h_rnd5-921_first_24h
)

for d in "${data_dirs[@]}"; do
    python3 ../fms_ehrs/scripts/aggregate_version_preds.py \
        --data_dir "$d" \
        --data_versions "${versions[@]}" \
        --model_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
        --out_dir "${hm}/figs"
done
