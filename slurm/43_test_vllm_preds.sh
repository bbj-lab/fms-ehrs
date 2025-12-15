#!/bin/bash

#SBATCH --job-name=m1_m2
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpudev
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB

source preamble.sh

python3 ../fms_ehrs/scripts/vllm_m1_m2.py \
    --data_dir "${hm}/data-mimic" \
    --data_version "W++_first_24h_llama-med-60358922_1-hp-W++_none_10pct" \
    --model_loc "${hm}/mdls-archive/llama-med-60358922_1-hp-W++" \
    --max_len 100000 \
    --n_samp 10 \
    --test_size 100

source postscript.sh
