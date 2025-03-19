#!/bin/bash

#SBATCH --job-name=tokenize-data
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=1TB
#SBATCH --time=1:00:00

source preamble.sh
python3 "${name}.py" \
    --data_dir "/scratch/$(whoami)/clif-data" \
    --data_version day_stays_qc \
    --vocab_path "${hm}/clif-data/day_stays_qc-tokenized/train/vocab.gzip" \
    --max_padded_len 1024 \
    --day_stay_filter true \
    --include_24h_cut true
