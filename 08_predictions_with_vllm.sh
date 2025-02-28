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
export hm=/gpfs/data/bbj-lab/users/burkh4rt
python3 08_predictions_with_vllm.py \
    --rep "${SLURM_ARRAY_TASK_ID}" \
    --data_dir ${hm}/clif-data/day_stays_qc_first_24h-tokenized \
    --model_dir ${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630 \
    --k 25000 \
    --n_samp 20 \
    --top_p 0.95
