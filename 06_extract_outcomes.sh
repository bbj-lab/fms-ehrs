#!/bin/bash

#SBATCH --job-name=extract-outcomes
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh
python3 "${name}.py" \
    --data_dir "/scratch/$(whoami)/clif-data" \
    --ref_version "day_stays_qc" \
    --data_version "day_stays_qc_first_24h"
