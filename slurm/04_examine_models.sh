#!/bin/bash

#SBATCH --job-name=examine-mdls
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/examine_models.py \
    --projector_type PCA \
    --data_dir "${hm}/clif-data" \
    --data_version "${data_version:-W++}" \
    --ref_mdl_loc "${hm}/clif-mdls-archive/llama-med-60358922_1-hp-W++" \
    --out_dir "${hm}/figs"
