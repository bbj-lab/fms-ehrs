#!/bin/bash

#SBATCH --job-name=extract-states
#SBATCH --output=./output/%j.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

source preamble.sh

torchrun --nproc_per_node=4 "${name}.py" \
    --data_dir "/scratch/$(whoami)/clif-data" \
    --data_version day_stays_qc_first_24h \
    --model_loc "${hm}/clif-mdls-archive/mdl-day_stays_qc-57350630" \
    --batch_sz $((2 ** 5))
