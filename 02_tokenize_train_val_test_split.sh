#!/bin/bash

#SBATCH --job-name=tokenize-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh
python3 "${name}.py" \
    --data_dir "/scratch/$(whoami)/clif-data" \
    --data_version_in QC \
    --data_version_out QC_day_stays \
    --vocab_path "${hm}/clif-data/day_stays_qc-tokenized/train/vocab.gzip" \
    --max_padded_len 1024 \
    --day_stay_filter true \
    --include_24h_cut true \
    --valid_admission_window "('2020-03-01','2022-03-01')"
