#!/bin/bash

#SBATCH --job-name=examine-mdls
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

source preamble.sh

python3 ../src/scripts/examine_models.py \
    --projector_type PCA \
    --data_dir "${hm}/clif-data" \
    --data_version QC_day_stays \
    --ref_mdl_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
    --out_dir "${hm}"
