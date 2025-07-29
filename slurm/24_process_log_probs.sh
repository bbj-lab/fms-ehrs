#!/bin/bash

#SBATCH --job-name=proc-log-probs
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

models=(
    llama-original-60358922_0-hp-W++
)
samp_orig=(
    "20826893"
    "27726633"
    "26624012"
    "24410460"
    "29173149"
    "24640534"
    "29022625"
    "27267707"
    "26886976"
)
samp_new=(
    "27055120"
    "792481"
    "12680680"
    "9468768"
    "8797520"
    "5296451"
    "10969205"
    "2974992"
    "20528107"
)

for m in "${models[@]}"; do
    for agg in sum perplexity; do
        python3 ../fms_ehrs/scripts/process_log_probs.py \
            --data_dir_orig "${hm}/data-mimic" \
            --data_dir_new "${hm}/data-ucmc" \
            --data_version "${m##*-}_first_24h" \
            --model_loc "${hm}/clif-mdls-archive/$m" \
            --out_dir "${hm}/figs" \
            --aggregation "${agg}" \
            --samp_orig "${samp_orig[@]}" \
            --samp_new "${samp_new[@]}"
    done
done
