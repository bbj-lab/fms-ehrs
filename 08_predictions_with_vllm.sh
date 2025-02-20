#!/bin/bash

#SBATCH --job-name=generate-predictions-vllm
#SBATCH --output=./output/%j.stdout
#SBATCH --chdir=/gpfs/data/bbj-lab/users/burkh4rt/clif-tokenizer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-19

source ~/.bashrc
source venv/bin/activate
python3 08_predictions_with_vllm.py --rep "${SLURM_ARRAY_TASK_ID}"
