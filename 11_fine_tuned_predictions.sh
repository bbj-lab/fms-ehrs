#!/bin/bash

#SBATCH --job-name=eval-ft-mdl
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00

hm="/gpfs/data/bbj-lab/users/$(whoami)"
cd "${hm}/clif-tokenizer" || exit
source ~/.bashrc
source venv/bin/activate
python3 11_fine_tuned_predictions.py \
    --data_dir "${hm}/clif-data/day_stays_qc_first_24h-tokenized" \
    --model_dir "${hm}/clif-mdls-archive/mdl-llama1b-sft-57451707-clsfr"
