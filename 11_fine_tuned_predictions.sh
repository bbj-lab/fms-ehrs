#!/bin/bash

#SBATCH --job-name=eval-ft-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source preamble.sh

#echo "fine-tuned mimic preds..."
#python3 "${name}.py" \
#    --data_dir "${hm}/clif-data/day_stays_qc_first_24h-tokenized" \
#    --model_dir "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630-57723914-clsfr"

echo "fine-tuned chicago preds..."
python3 "${name}.py" \
    --data_dir "/scratch/burkh4rt/clif-data/day_stays_qc_first_24h-tokenized" \
    --model_dir "${hm}/clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630-57723914-clsfr"
