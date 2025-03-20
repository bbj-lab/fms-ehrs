#!/bin/bash

#SBATCH --job-name=partition-data
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=1TB
#SBATCH --time=1:00:00

source preamble.sh

python3 "${name}.py" \
    --data_dir_in "/scratch/$(whoami)/CLIF-2.0.0" \
    --data_dir_out "/scratch/$(whoami)/clif-data" \
    --train_frac 0.05 \
    --val_frac 0.05
