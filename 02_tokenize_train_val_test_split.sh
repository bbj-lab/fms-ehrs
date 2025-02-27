#!/bin/bash

#SBATCH --job-name=tokenize-data
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

hm="/gpfs/data/bbj-lab/users/$(whoami)"
cd "${hm}/clif-tokenizer" || exit
source ~/.bashrc
source venv/bin/activate
python3 02_tokenize_train_val_test_split.py \
     --data_dir "${hm}/clif-data/" \
     --data_version day_stays_qc \
     --max_padded_len 1024 \
     --day_stay_filter true \
     --include_24h_cut true
