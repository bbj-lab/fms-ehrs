#!/bin/bash

#SBATCH --job-name=eval-ft-mdl
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00

source ~/.bashrc
source venv/bin/activate
export hm=/gpfs/data/bbj-lab/users/burkh4rt
python3 11_fine_tuned_predictions.py \
    --data_dir ${hm}/clif-data/day_stays_qc_first_24h-tokenized \
    --model_dir ${hm}/clif-mdls-archive/mdl-llama1b-sft-57451707-clsfr
