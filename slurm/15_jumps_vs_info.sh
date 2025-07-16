#!/bin/bash

#SBATCH --job-name=jumps-inf
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --time=24:00:00

source preamble.sh

model=llama-med-60358922_1-hp-W++
data_dirs=("${hm}/clif-data" "${hm}/clif-data-ucmc")

for d in "${data_dirs[@]}"; do
    echo "Generating plots..."
    python3 ../fms_ehrs/scripts/process_rep_trajs_inf.py \
        --data_dir "$d" \
        --data_versions "${model##*-}" \
        --model_loc "${hm}/clif-mdls-archive/${model}" \
        --out_dir "${hm}/figs" \
        --make_plots \
        --aggregation "sum" \
        --drop_prefix \
        --skip_kde
done
