#!/bin/bash

#SBATCH --job-name=proc-info
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --time=1:00:00

source preamble.sh

model=llama-med-60358922_1-hp-W++
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

echo "Processing timelines..."
python3 ../fms_ehrs/scripts/process_log_probs.py \
    --data_dir_orig "${hm}/data-mimic" \
    --data_dir_new "${hm}/data-ucmc" \
    --data_version "${model##*-}" \
    --model_loc "${hm}/mdls-archive/${model}" \
    --out_dir "${hm}/figs" \
    --aggregation sum \
    --samp_orig "${samp_orig[@]}" \
    --samp_new "${samp_new[@]}" \
    --n_egs 10 \
    --max_len 102
