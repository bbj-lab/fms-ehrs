#!/bin/bash

#SBATCH --job-name=generate-predictions-vllm
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-19

hm="/gpfs/data/bbj-lab/users/$(whoami)"
cd "${hm}/clif-tokenizer" || exit
source ~/.bashrc
source venv/bin/activate
python3 08_predictions_with_vllm.py \
    --rep "${SLURM_ARRAY_TASK_ID}" \
    --data_dir "${hm}/clif-data/day_stays_qc_first_24h-tokenized" \
    --model_dir "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630" \
    --k 25000 \
    --n_samp 20 \
    --top_p 0.95
