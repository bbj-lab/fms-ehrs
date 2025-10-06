#!/bin/bash

#SBATCH --job-name=xtract-attns
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpudev
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

source preamble.sh

for opt in "" "--final_layer"; do
    python3 ../fms_ehrs/scripts/extract_attentions.py \
        --data_dir "../../data-mimic" \
        --data_version_out "W++" \
        --model_loc "${hm}/mdls-archive/llama-med-60358922_1-hp-W++" \
        $opt
done
