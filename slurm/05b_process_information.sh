#!/bin/bash

#SBATCH --job-name=proc-info
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --time=1:00:00

source preamble.sh

samp=(
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

python3 ../fms_ehrs/scripts/process_information.py \
    --data_dir "${hm}/data-mimic" \
    --data_version "V21" \
    --model_loc "${hm}/mdls-archive/llama-med-4476655-hp-V21" \
    --out_dir "${hm}/figs" \
    --samp "${samp[@]}" \
    --emit_json True \
    --max_len 300

source postscript.sh
