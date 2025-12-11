#!/bin/bash

#SBATCH --job-name=gen-comp
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpudev
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

source preamble.sh

python3 ../fms_ehrs/scripts/generate_all_completions.py \
    --data_dir "../../data-mimic" \
    --data_version "W++_first_24h_llama-med-60358922_1-hp-W++_none_10pct_ppy" \
    --model_loc "${hm}/mdls-archive/llama-med-60358922_1-hp-W++" \
    --max_len 10000 \
    --n_samp 20

source postscript.sh
