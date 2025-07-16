#!/bin/bash

#SBATCH --job-name=pull-mimic
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
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

echo "Pulling info on MIMIC hospitalizations..."
python3 ../fms_ehrs/scripts/query_raw_mimic.py \
    --data_dir "${hm}/mimiciv-3.1" \
    --hadm_ids "${samp[@]}"
