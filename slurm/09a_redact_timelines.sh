#!/bin/bash

#SBATCH --job-name=redact-tls
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00
#SBATCH --array=0-47

source preamble.sh

ni=2
nj=6
nk=4
nm=1
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
    # abs-gmm-same_admission_death
    # abs-imp-same_admission_death
    # rel-gmm-same_admission_death
    # rel-imp-same_admission_death
    # abs-gmm-long_length_of_stay
    # abs-imp-long_length_of_stay
    # rel-gmm-long_length_of_stay
    # rel-imp-long_length_of_stay
    importance-h2o-mean
)

python3 ../fms_ehrs/scripts/redact_timelines_tokenwise.py \
    --data_dir "${data_dirs[$i]}" \
    --data_version "Y21_icu24_first_24h" \
    --model_loc "${hm}/mdls-archive/gemma-5635921-Y21" \
    --pct "${pcts[$k]}" \
    --method "${methods[$j]}" \
    --metric "${metrics[$m]}"
# --x_infm

source postscript.sh
