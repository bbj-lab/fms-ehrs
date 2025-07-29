#!/bin/bash

#SBATCH --job-name=jumps-inf
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --time=24:00:00

source preamble.sh

data_dirs=(
<<<<<<< HEAD
    "${hm}/clif-data"
    # "${hm}/clif-data-ucmc"
=======
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
>>>>>>> main
)

for d in "${data_dirs[@]}"; do
    python3 ../fms_ehrs/scripts/process_rep_trajs_inf.py \
        --data_dir "$d" \
        --data_versions "W++" \
        --model_loc "${hm}/clif-mdls-archive/llama-med-60358922_1-hp-W++" \
        --out_dir "${hm}/figs" \
        --make_plots \
        --aggregation "sum" \
        --drop_prefix \
        --skip_kde
done
