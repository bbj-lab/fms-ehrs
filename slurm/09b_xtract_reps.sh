#!/bin/bash

#SBATCH --job-name=xtract-reps
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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

data_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
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
    abs-gmm-same_admission_death
    abs-imp-same_admission_death
    rel-gmm-same_admission_death
    rel-imp-same_admission_death
    abs-gmm-long_length_of_stay
    abs-imp-long_length_of_stay
    rel-gmm-long_length_of_stay
    rel-imp-long_length_of_stay
)

mdl=gemma-5635921-Y21
tgt="Y21_icu24_red_${metrics[$m]}-x-infm_${methods[$j]}${pcts[$k]}pct-${mdl}_first_24h"

python3 ../fms_ehrs/scripts/extract_hidden_states.py \
    --data_dir "${data_dirs[$i]}" \
    --data_version "${tgt}" \
    --model_loc "${hm}/mdls-archive/${mdl}" \
    --batch_sz $((2 ** 5))

source postscript.sh
