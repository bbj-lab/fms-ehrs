#!/bin/bash

#SBATCH --job-name=extract-outcomes
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source ~/.bashrc
source venv/bin/activate
python3 06_extract_outcomes.py \
    --ref_version "day_stays_qc" \
    --data_version "day_stays_qc_first_24h"
