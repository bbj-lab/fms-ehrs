#!/bin/bash

#SBATCH --job-name=run-extub
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/run_successful_extubation_determination.py \
    --data_dir "${hm}/data-raw/ucmc-2.1.0" \
    --q_dir "${hm}/fms-ehrs-reps/fms_ehrs/misc" \
    --tz "US/Central"

python3 ../fms_ehrs/scripts/run_successful_extubation_determination.py \
    --data_dir "${hm}/data-raw/mimic-2.1.0" \
    --q_dir "${hm}/fms-ehrs-reps/fms_ehrs/misc" \
    --tz "US/Eastern"

source postscript.sh
