#!/bin/bash

#SBATCH --job-name=cluster-reps
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00

source preamble.sh

m=llama1b-smol-59946181-hp-QC_noX

python3 ../fms_ehrs/scripts/cluster_reps.py \
    --data_dir_orig "${hm}/data-mimic" \
    --data_dir_new "${hm}/data-ucmc" \
    --data_version "${m##*-}_first_24h" \
    --model_loc "${hm}/clif-mdls-archive/$m"
