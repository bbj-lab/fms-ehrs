#!/bin/bash

#SBATCH --job-name=add-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

python3 ../src/scripts/add_posthoc_data.py \
    --new_data_loc "${hm}/clif-data/scratch/machine_measurements.csv" \
    --data_version QC \
    --patient_id_col subject_id \
    --time_col ecg_time
