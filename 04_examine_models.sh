#!/bin/bash

#SBATCH --job-name=examine-mdls
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

source preamble.sh

python3 "${name}.py" \
    --projector_type PCA \
    --data_dir "${hm}/clif-data" \
    --data_version QC_day_stays \
    --ref_mdl_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
    --out_dir "${hm}"

#    --addl_mdls_loc "${hm}/clif-mdls-archive/mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death,\
#                     ${hm}/clif-mdls-archive/mdl-llama1b-57928921-run1-58134628-clsfr-long_length_of_stay,\
#                     ${hm}/clif-mdls-archive/mdl-llama1b-57928921-run1-58165531-clsfr-imv_event,\
#                     ${hm}/clif-mdls-archive/mdl-llama1b-57928921-run1-58165534-clsfr-icu_admission" \
