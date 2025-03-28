#!/bin/bash

#SBATCH --job-name=all-states
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=0
#SBATCH --time=24:00:00

hm="/gpfs/data/bbj-lab/users/$(whoami)"
cd "${hm}/clif-tokenizer" || exit
source ~/.bashrc
source venv/bin/activate
python3 14_process_representation_trajectories.py \
    --data_dir "${hm}/clif-data" \
    --data_version day_stays_qc_first_24h \
    --model_loc "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630" \
    --save_jumps True
