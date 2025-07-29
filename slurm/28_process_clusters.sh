#!/bin/bash

#SBATCH --job-name=proc-clusters
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00

source preamble.sh

m=llama1b-smol-59946181-hp-QC_noX

python3 ../fms_ehrs/scripts/process_clusters.py \
    --data_dir_orig "${hm}/data-mimic" \
    --data_dir_new "${hm}/data-ucmc" \
    --data_version "${m##*-}_first_24h" \
    --model_loc "${hm}/mdls-archive/$m"
