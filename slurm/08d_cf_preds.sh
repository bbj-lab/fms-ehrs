#!/bin/bash

#SBATCH --job-name=cf-perf-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=5
#SBATCH --time=1:00:00

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
    abs-gmm-same_admission_death
    abs-imp-same_admission_death
    rel-gmm-same_admission_death
    rel-imp-same_admission_death
    abs-gmm-long_length_of_stay
    abs-imp-long_length_of_stay
    rel-gmm-long_length_of_stay
    rel-imp-long_length_of_stay
)
outcomes=(
    "same_admission_death"
    "long_length_of_stay"
    "ama_discharge"
    "hospice_discharge"
)

mdl=gemma-5635921-Y21

for d in "${data_dirs[@]}"; do
    versions=("Y21_icu24_red_information_none10pct-${mdl}_first_24h")
    handles=("original")
    for metric in "${metrics[@]}"; do
        for method in "${methods[@]:1}"; do
            for pct in "${pcts[@]}"; do
                versions+=("Y21_icu24_red_${metric}_${method}${pct}pct-${mdl}_first_24h")
                handles+=("${metric}_${method}${pct}pct")
            done
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
        --outcomes "${outcomes[@]}"

done
