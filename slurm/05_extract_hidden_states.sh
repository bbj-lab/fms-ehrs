#!/bin/bash

#SBATCH --job-name=extract-states
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --array=0-31

source preamble.sh

ni=2
nj=4
nk=4
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
    20
    30
    40
)

models=(
    llama-original-60358922_0-hp-W++
    llama-med-60358922_1-hp-W++
    llama-small-60358922_2-hp-W++
    llama-smol-60358922_3-hp-W++
)

torchrun --nproc_per_node=4 \
    --rdzv_backend c10d \
    --rdzv-id "$SLURM_ARRAY_TASK_ID" \
    --rdzv-endpoint=localhost:0 \
    ../fms_ehrs/scripts/extract_hidden_states.py \
    --data_dir "${data_dirs[$i]}" \
    --data_version "W++_first_24h_${models[$k]}_${methods[$j]}_20pct" \
    --model_loc "${hm}/clif-mdls-archive/${models[$k]}" \
    --batch_sz $((2 ** 5))
