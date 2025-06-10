#!/bin/bash

#SBATCH --job-name=reduce-tls
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00
#SBATCH --array=0-7

source preamble.sh

div=2
quo=$((SLURM_ARRAY_TASK_ID / div))
rem=$((SLURM_ARRAY_TASK_ID % div))

data_dirs=(
    "${hm}/clif-data"
    "${hm}/clif-data-ucmc"
)
methods=(
    none
    top_k
    bottom_k
    random_k
)
new_versions=(
    icu24h_first_24h
    icu24h_top5-921_first_24h
    icu24h_bot5-921_first_24h
    icu24h_rnd5-921_first_24h
)

python3 ../fms_ehrs/scripts/redact_timelines.py \
    --data_dir "${data_dirs[$rem]}" \
    --data_version "QC_day_stays_first_24h" \
    --model_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
    --k 5 \
    --method "${methods[$quo]}" \
    --new_version "${new_versions[$quo]}" \
    --aggregation sum
