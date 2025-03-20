#!/bin/bash

#SBATCH --job-name=transfer-rep-preds
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=100GB
#SBATCH --time=2:00:00

source preamble.sh

python3 "${name}.py"
