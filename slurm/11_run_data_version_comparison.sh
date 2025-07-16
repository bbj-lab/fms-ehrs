#!/bin/bash

#SBATCH --job-name=cf-perf-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=5
#SBATCH --time=1:00:00

source preamble.sh

model=llama-med-60358922_1-hp-W++
data_dirs=("${hm}/clif-data" "${hm}/clif-data-ucmc")
methods=(none top bottom random)
pcts=(10 20 30 40)

for d in "${data_dirs[@]}"; do
    versions=("W++_first_24h_${model}_none_20pct")
    handles=("original")
    for me in "${methods[@]:1}"; do
        for p in "${pcts[@]}"; do
            versions+=("W++_first_24h_${model}_${me}_${p}pct")
            handles+=("${me}_${p}pct")
        done
    done

    echo "Comparing performance across data versions..."
    python3 ../fms_ehrs/scripts/aggregate_version_preds.py \
        --data_dir "$d" \
        --data_versions "${versions[@]}" \
        --handles "${handles[@]}" \
        --baseline_handle none \
        --model_loc "${hm}/clif-mdls-archive/${model}" \
        --out_dir "${hm}/figs"

done
