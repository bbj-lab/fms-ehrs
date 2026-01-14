#!/bin/bash

#SBATCH --job-name=proc-imps
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=tier3q
#SBATCH --time=24:00:00

source preamble.sh

splits=("test")
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
    python3 ../fms_ehrs/scripts/process_all_importances.py \
        --data_dir "$ddir" \
        --data_version V21 \
        --model_loc "${hm}/mdls-archive/llama-med-4476655-hp-V21" \
        --metrics "${metrics[@]}" \
        --splits "${splits[@]}"
done
