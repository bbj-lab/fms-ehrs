#!/bin/bash

#SBATCH --job-name=redact-tls
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1:00:00
#SBATCH --array=0-23

source preamble.sh

ni=2
nj=4
nk=3
i=$((SLURM_ARRAY_TASK_ID % ni))
jk=$((SLURM_ARRAY_TASK_ID / ni))
j=$((jk % nj))
k=$((jk / nj))

if ((SLURM_ARRAY_TASK_COUNT != ni * nj * nk)); then
    echo "Warning:"
    echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"
    echo "ni*nj*nk=$((ni * nj * nk))"
fi

data_dirs=(
    "${hm}/clif-data"
    "${hm}/clif-data-ucmc"
)
methods=(
    none
    top
    bottom
    random
)
pcts=(
    10
    30
    40
)
fracs=(0.1 0.3 0.4)

python3 ../fms_ehrs/scripts/redact_timelines.py \
    --data_dir "${data_dirs[$i]}" \
    --data_version "W++_first_24h" \
    --model_loc "${hm}/clif-mdls-archive/llama-med-60358922_1-hp-W++" \
    --pct "${fracs[$k]}" \
    --method "${methods[$j]}" \
    --new_version "W++_first_24h_llama-med-60358922_1-hp-W++_${methods[$j]}_${pcts[$k]}pct" \
    --aggregation sum
