#!/bin/bash

#SBATCH --job-name=xfer-reps
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=20GB
#SBATCH --time=4:00:00
#SBATCH --array=0-215

source preamble.sh

ni=6
nj=4
nk=9
i=$((SLURM_ARRAY_TASK_ID % ni))
jk=$((SLURM_ARRAY_TASK_ID / ni))
j=$((jk % nj))
k=$((jk / nj))

if ((SLURM_ARRAY_TASK_COUNT != ni * nj * nk)); then
    echo "Warning:"
    echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"
    echo "ni*nj*nk=$((ni * nj * nk))"
fi

methods=(
    none
    top
    bottom
    random
    top_abs
    btm_abs
)
pcts=(
    10
    20
    30
    40
)
metrics=(
    information
    abs-gmm-same_admission_death
    abs-imp-same_admission_death
    rel-gmm-same_admission_death
    rel-imp-same_admission_death
    abs-gmm-long_length_of_stay
    abs-imp-long_length_of_stay
    rel-gmm-long_length_of_stay
    rel-imp-long_length_of_stay
)
outcomes=(
    "same_admission_death"
    "long_length_of_stay"
    "ama_discharge"
    "hospice_discharge"
)

mdl=gemma-5635921-Y21
tgt="Y21_icu24_red_${metrics[$k]}_${methods[$i]}${pcts[$j]}pct-${mdl}_first_24h"

python3 ../fms_ehrs/scripts/transfer_rep_based_preds.py \
    --data_dir_orig "${hm}/data-mimic" \
    --data_dir_new "${hm}/data-ucmc" \
    --data_version "${tgt}" \
    --model_loc "${hm}/mdls-archive/${mdl}" \
    --classifier light_gbm \
    --outcomes "${outcomes[@]}" \
    --save_preds

source postscript.sh
