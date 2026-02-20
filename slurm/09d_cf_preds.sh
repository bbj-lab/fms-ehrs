#!/bin/bash

#SBATCH --job-name=cf-perf-data
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=5
#SBATCH --time=4:00:00
#SBATCH --array=0-8

source preamble.sh

data_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
)
methods=(
    none
    top
    bottom
    random
    top_abs
    btm_abs
)
pcts=(
    10
    20
    30
    40
)
metrics=(
    information
    abs-gmm-same_admission_death-x-infm
    abs-imp-same_admission_death-x-infm
    rel-gmm-same_admission_death-x-infm
    rel-imp-same_admission_death-x-infm
    abs-gmm-long_length_of_stay-x-infm
    abs-imp-long_length_of_stay-x-infm
    rel-gmm-long_length_of_stay-x-infm
    rel-imp-long_length_of_stay-x-infm
)
outcomes=(
    "same_admission_death"
    "long_length_of_stay"
    # "ama_discharge"
    # "hospice_discharge"
)

mdl=gemma-5635921-Y21
metric="${metrics[$SLURM_ARRAY_TASK_ID]}"

for d in "${data_dirs[@]}"; do
    versions=("Y21_icu24_red_information_none10pct-${mdl}_first_24h")
    handles=("original")

    for method in "${methods[@]:1}"; do
        for pct in "${pcts[@]}"; do
            versions+=("Y21_icu24_red_${metric}_${method}${pct}pct-${mdl}_first_24h")
            handles+=("${metric}_${method}${pct}pct")
        done
    done

    echo "Comparing performance across data versions..."
    python3 ../fms_ehrs/scripts/aggregate_version_preds.py \
        --data_dir "$d" \
        --data_versions "${versions[@]}" \
        --handles "${handles[@]}" \
        --baseline_handle "original" \
        --model_loc "${hm}/mdls-archive/${mdl}" \
        --out_dir "${hm}/figs" \
        --outcomes "${outcomes[@]}" \
        --classifier light_gbm \
        --suffix light_gbm \
        --n_bootstrap_samples 1000
done

source postscript.sh
