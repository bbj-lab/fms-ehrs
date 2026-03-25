#!/bin/bash

#SBATCH --job-name=investigate-weights
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh

data_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
)

for d in "${data_dirs[@]}"; do
    python3 ../fms_ehrs/scripts/investigate_weights.py \
        --data_dir "$d" \
        --data_version_out "W++" \
        --model_loc "${hm}/mdls-archive/llama-med-60358922_1-hp-W++" \
        --max_len 300 \
        --save_plots
done
