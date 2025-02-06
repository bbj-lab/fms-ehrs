#!/bin/bash

#SBATCH --job-name=generate-mamba-clif
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source ~/.bashrc
source venv/bin/activate
python3 8_run_autoregressive_predictions.py
