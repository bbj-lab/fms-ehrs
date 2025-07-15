#!/bin/bash

#SBATCH --job-name=tokenize-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh

[ -z "${data_version}" ] && export data_version=W++

echo "Processing MIMIC data..."
python3 ../fms_ehrs/scripts/tokenize_train_val_test_split.py \
    --data_dir "${hm}/clif-data/" \
    --data_version_in raw \
    --data_version_out "${data_version}" \
    --max_padded_len 1024 \
    --day_stay_filter True \
    --include_24h_cut True \
    --drop_nulls_nans True

echo "Using vocab from MIMIC to process UCMC data..."
python3 ../fms_ehrs/scripts/tokenize_train_val_test_split.py \
    --data_dir "${hm}/clif-data-ucmc" \
    --data_version_in raw \
    --data_version_out "${data_version}" \
    --vocab_path "${hm}/clif-data/${data_version}-tokenized/train/vocab.gzip" \
    --max_padded_len 1024 \
    --day_stay_filter True \
    --include_24h_cut True \
    --valid_admission_window "('2020-03-01','2022-03-01')" \
    --drop_nulls_nans True
