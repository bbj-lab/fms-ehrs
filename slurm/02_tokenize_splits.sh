#!/bin/bash

#SBATCH --job-name=tokenize-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

source preamble.sh

export data_version=W21

echo "Processing MIMIC data..."
python3 ../fms_ehrs/scripts/tokenize_w_config.py \
    --data_dir "${hm}/data-mimic/" \
    --data_version_in W21 \
    --data_version_out ${data_version} \
    --include_24h_cut \
    --config_loc "../fms_ehrs/config/config-21.yaml"

echo "Using vocab from MIMIC to process UCMC..."
python3 ../fms_ehrs/scripts/tokenize_w_config.py \
    --data_dir "${hm}/data-ucmc" \
    --vocab_path "${hm}/data-mimic/${data_version}-tokenized/train/vocab.gzip" \
    --data_version_in W21 \
    --data_version_out ${data_version} \
    --include_24h_cut \
    --config_loc "../fms_ehrs/config/config-21.yaml"
