#!/bin/bash

#SBATCH --job-name=run-clifpy
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/run_clifpy.py \
    --data_dir "${hm}/data-raw/ucmc-2.1.0" \
    --out_dir "${hm}/figs" \
    --tz "US/Central" \
    --waterfall

python3 ../fms_ehrs/scripts/run_clifpy.py \
    --data_dir "${hm}/data-raw/mimic-2.1.0" \
    --out_dir "${hm}/figs" \
    --tz "US/Eastern" \
    --waterfall

source postscript.sh
