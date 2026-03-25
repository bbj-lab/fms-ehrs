#!/bin/bash

#SBATCH --job-name=proc-attns
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

splits=("train" "val" "test")
data_dirs=("${hm}/data-mimic" "${hm}/data-ucmc")
metrics=(
    "h2o-mean"
    "h2o-mean_log"
    "h2o-va-mean"
    "h2o-va-mean_log"
    "scissorhands-10"
    "scissorhands-20"
    "scissorhands-va-10"
    "scissorhands-va-20"
    "rollout-mean"
    "rollout-mean_log"
    "h2o-normed-mean"
    "h2o-normed-mean_log"
)

for ddir in "${data_dirs[@]}"; do
    python3 ../fms_ehrs/scripts/process_extracted_attentions.py \
        --data_dir "$ddir" \
        --data_version "W++" \
        --model_loc "${hm}/mdls-archive/llama-med-60358922_1-hp-W++" \
        --metrics "${metrics[@]}" \
        --splits "${splits[@]}"
done
