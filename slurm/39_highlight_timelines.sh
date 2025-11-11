#!/bin/bash

#SBATCH --job-name=hlt-tls
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

metrics=(
    "h2o-mean"
    "h2o-mean_log"
    "h2o-va-mean"
    "h2o-va-mean_log"
    #    "scissorhands-10"
    #    "scissorhands-20"
    #    "scissorhands-va-10"
    #    "scissorhands-va-20"
    "rollout-mean"
    "rollout-mean_log"
    "h2o-normed-mean"
    "h2o-normed-mean_log"
)
ids=(
    "24640534" # cf. Fig. 2
    "26886976" # Fig. 3
    "29022625" # Fig. 4
    #    "20606203"
    #    "29298288"
    #    "28910506"
    #    "28812737"
    #    "20606203"
    #    "29866426"
)

python3 ../fms_ehrs/scripts/visualize_timelines.py \
    --data_dir "${hm}/data-mimic" \
    --data_version "W++" \
    --model_loc "${hm}/mdls-archive/llama-med-60358922_1-hp-W++" \
    --metrics "${metrics[@]}" \
    --ids "${ids[@]}" \
    --out_dir "${hm}/figs" \
    --tl_len 1024

models=(
    "llama-med-60358922_1-hp-W++-sft-mort"
    "llama-med-60358922_1-hp-W++-sft-llos"
    "llama-med-60358922_1-hp-W++-sft-icua"
    "llama-med-60358922_1-hp-W++-sft-imve"
)
metrics=(
    "saliency"
    "smoothgrad-scaled"
)
for mdl in "${models[@]}"; do
    python3 ../fms_ehrs/scripts/visualize_timelines.py \
        --data_dir "${hm}/data-mimic" \
        --data_version "W++_first_24h" \
        --model_loc "${hm}/mdls-archive/${mdl}" \
        --metrics "${metrics[@]}" \
        --ids "${ids[@]}" \
        --out_dir "${hm}/figs" \
        --tl_len 1024
done
