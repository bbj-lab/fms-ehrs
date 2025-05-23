#!/bin/bash

# download and process MIMIC data into CLIF format with this script

pip install jq
git clone git@github.com:Common-Longitudinal-ICU-data-Format/CLIF-MIMIC.git \
    CLIF-MIMICv0.1.0 --branch v0.1.0
cd CLIF-MIMICv0.1.0 || exit
wget -r -N -c -np --user "$(whoami)" --ask-password https://physionet.org/files/mimiciv/3.1/
cat <<< "$(jq --arg path "$(pwd)/physionet.org/files/mimiciv/3.1/" \
    '.default.mimic_csv_dir = $path' ./config/config_template.json)" \
> ./config/config.json
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py

# this leaves parquet tables in ./output/rclif-2.0
