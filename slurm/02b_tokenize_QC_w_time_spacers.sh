#!/bin/bash

#SBATCH --job-name=tokenize-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=100GB
#SBATCH --time=24:00:00

source preamble.sh

[ -z "${data_version}" ] && export data_version=QC_spaced

echo "Processing MIMIC data..."
python3 ../fms_ehrs/scripts/tokenize_train_val_test_split.py \
    --data_dir "${hm}/data-mimic/" \
    --data_version_in QC \
    --data_version_out "${data_version:-QC_noX}" \
    --max_padded_len 1024 \
    --day_stay_filter True \
    --include_24h_cut True \
    --drop_nulls_nans True \
    --include_time_spacing_tokens True

echo "Using vocab from MIMIC to process UCMC data..."
python3 ../fms_ehrs/scripts/tokenize_train_val_test_split.py \
    --data_dir "${hm}/data-ucmc" \
    --data_version_in QC \
    --data_version_out "${data_version:-QC_noX}" \
    --vocab_path "${hm}/data-mimic/${data_version:-QC_noX}-tokenized/train/vocab.gzip" \
    --max_padded_len 1024 \
    --day_stay_filter True \
    --include_24h_cut True \
    --valid_admission_window "('2020-03-01','2022-03-01')" \
    --drop_nulls_nans True \
    --include_time_spacing_tokens True
