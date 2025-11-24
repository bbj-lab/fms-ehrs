#!/bin/bash

#SBATCH --job-name=run-sofa
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=300GB
#SBATCH --time=1:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/run_sofa_scoring.py \
    --data_dir "${hm}/data-raw/ucmc-2.1.0" \
    --tz "US/Central"

python3 ../fms_ehrs/scripts/run_sofa_scoring.py \
    --data_dir "${hm}/data-raw/mimic-2.1.0" \
    --tz "US/Eastern"

source postscript.sh
