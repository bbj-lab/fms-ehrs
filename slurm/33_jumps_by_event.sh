#!/bin/bash

#SBATCH --job-name=jumps-ev
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --cpus-per-task=5
#SBATCH --time=1:00:00

source preamble.sh

data_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
)

for d in "${data_dirs[@]}"; do
    python3 ../fms_ehrs/scripts/rep_changes_by_event.py \
        --data_dir "$d" \
        --data_version "W++_first_24h" \
        --model_loc "${hm}/clif-mdls-archive/llama-med-60358922_1-hp-W++" \
        --aggregation "sum" \
        --big_batch_sz $((2 ** 12)) \
        --out_dir "${hm}/figs"
done

# NB: big_batch_sz needs to match big_batch_sz used in extract_all_hidden_states
