#!/bin/bash

#SBATCH --job-name=xtract-attns
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpudev
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

source preamble.sh

aggs=(sum mean max median Q75 Q90 Q95 mean_log median_log)
ids=("24640534" "26886976" "29022625")

#python3 ../fms_ehrs/scripts/extract_attentions.py \
#    --data_dir "../../data-mimic" \
#    --data_version "W++" \
#    --model_loc "${hm}/mdls-archive/llama-med-60358922_1-hp-W++" \
#    --max_len 210 \
#    --ids "${ids[@]}" \
#    --agg_fns "${aggs[@]}" \
#    --drop_labels

python3 ../fms_ehrs/scripts/extract_attentions.py \
    --data_dir "../../data-mimic" \
    --data_version "QC_day_stays" \
    --model_loc "${hm}/mdls-archive/llama1b-57928921-run1" \
    --max_len 210 \
    --ids "${ids[@]}" \
    --agg_fns "${aggs[@]}" \
    --drop_labels
