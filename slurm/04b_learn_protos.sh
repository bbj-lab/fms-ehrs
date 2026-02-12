#!/bin/bash

#SBATCH --job-name=xtract-all
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

source preamble.sh

python3 ../fms_ehrs/scripts/learn_prototypes.py \
    --data_dir_orig "${hm}/data-mimic" \
    --data_dir_new "${hm}/data-ucmc" \
    --data_version Y21_icu24_first_24h \
    --model_loc "${hm}/mdls-archive/gemma-5635921-Y21" \
    --save_params

python3 ../fms_ehrs/scripts/learn_prototypes.py \
    --data_dir_orig "${hm}/data-mimic" \
    --data_dir_new "${hm}/data-ucmc" \
    --data_version Y21_unfused_icu24_first_24h \
    --model_loc "${hm}/mdls-archive/gemma-5687290-Y21_unfused" \
    --save_params

source postscript.sh
