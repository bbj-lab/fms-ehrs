#!/bin/bash

#SBATCH --job-name=examine-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=5:00

source preamble.sh

model=llama-med-60358922_1-hp-W++

echo "Examining learned embeddings..."
python3 ../fms_ehrs/scripts/examine_models.py \
    --projector_type PCA \
    --data_dir "${hm}/data-mimic" \
    --data_version "${model##*-}" \
    --ref_mdl_loc "${hm}/mdls-archive/${model}" \
    --out_dir "${hm}/figs"
