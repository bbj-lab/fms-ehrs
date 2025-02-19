#!/bin/bash

#SBATCH --job-name=tune-clif-mamba
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00

source ~/.bashrc
source venv/bin/activate
torchrun --nproc_per_node=8 03_tune_mamba_with_packing.py
