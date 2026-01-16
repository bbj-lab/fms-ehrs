#!/bin/bash

#SBATCH --job-name=run-clifpy
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=300GB
#SBATCH --time=3:00:00

source preamble.sh

# clifpy
python3 ../fms_ehrs/scripts/run_clifpy.py \
    --data_dir "${hm}/data-raw/ucmc-2.1.0" \
    --out_dir "${hm}/figs" \
    --tz "UTC" \
    --waterfall \
    --convert_doses_intermittent \
    --convert_doses_continuous \
    --validate

python3 ../fms_ehrs/scripts/run_clifpy.py \
    --data_dir "${hm}/data-raw/mimic-2.1.0" \
    --out_dir "${hm}/figs" \
    --tz "UTC" \
    --waterfall \
    --convert_doses_intermittent \
    --convert_doses_continuous \
    --validate

# extub
python3 ../fms_ehrs/scripts/run_successful_extubation_determination.py \
    --data_dir "${hm}/data-raw/ucmc-2.1.0" \
    --q_dir "${hm}/fms-ehrs-reps/fms_ehrs/misc" \
    --tz "UTC"

python3 ../fms_ehrs/scripts/run_successful_extubation_determination.py \
    --data_dir "${hm}/data-raw/mimic-2.1.0" \
    --q_dir "${hm}/fms-ehrs-reps/fms_ehrs/misc" \
    --tz "UTC"

# sofa
python3 ../fms_ehrs/scripts/run_sofa_scoring.py \
    --data_dir "${hm}/data-raw/ucmc-2.1.0" \
    --tz "UTC"

python3 ../fms_ehrs/scripts/run_sofa_scoring.py \
    --data_dir "${hm}/data-raw/mimic-2.1.0" \
    --tz "UTC"

source postscript.sh
