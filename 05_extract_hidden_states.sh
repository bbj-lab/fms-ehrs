#!/bin/bash

#SBATCH --job-name=extract-states
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

source ~/.bashrc
source venv/bin/activate
torchrun --nproc_per_node=8 05_extract_hidden_states.py
