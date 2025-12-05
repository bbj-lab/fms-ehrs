#!/bin/bash

#SBATCH --job-name=tokenize-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=500GB
#SBATCH --time=8-00:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/tokenize_meds.py \
    --data_dir "${hm}/data-raw/mimic-meds-ed-ihlee" \
    --config_loc "${hm}/fms-ehrs-reps/fms_ehrs/config/mimic-meds-ed.yaml"

source postscript.sh
