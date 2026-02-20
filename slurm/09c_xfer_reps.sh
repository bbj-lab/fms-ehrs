#!/bin/bash

#SBATCH --job-name=xfer-reps
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=20GB
#SBATCH --time=4:00:00
#SBATCH --array=0-383

source preamble.sh

ni=2
nj=6
nk=4
nm=8
i=$((SLURM_ARRAY_TASK_ID % ni))
jkm=$((SLURM_ARRAY_TASK_ID / ni))
j=$((jkm % nj))
km=$((jkm / nj))
k=$((km % nk))
m=$((km / nk))

if ((SLURM_ARRAY_TASK_COUNT != ni * nj * nk * nm)); then
    echo "Warning:"
    echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"
    echo "ni*nj*nk*nm=$((ni * nj * nk * nm))"
fi

classifiers=(
    logistic_regression
    light_gbm
)
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
    abs-gmm-same_admission_death-x-infm
    abs-imp-same_admission_death-x-infm
    rel-gmm-same_admission_death-x-infm
    rel-imp-same_admission_death-x-infm
    abs-gmm-long_length_of_stay-x-infm
    abs-imp-long_length_of_stay-x-infm
    rel-gmm-long_length_of_stay-x-infm
    rel-imp-long_length_of_stay-x-infm
)
outcomes=(
    "same_admission_death"
    "long_length_of_stay"
    "ama_discharge"
    "hospice_discharge"
)

mdl=gemma-5635921-Y21
tgt="Y21_icu24_red_${metrics[$m]}_${methods[$j]}${pcts[$k]}pct-${mdl}_first_24h"

python3 ../fms_ehrs/scripts/transfer_rep_based_preds.py \
    --data_dir_orig "${hm}/data-mimic" \
    --data_dir_new "${hm}/data-ucmc" \
    --data_version "${tgt}" \
    --model_loc "${hm}/mdls-archive/${mdl}" \
    --classifier "${classifiers[$i]}" \
    --outcomes "${outcomes[@]}" \
    --save_preds

source postscript.sh
