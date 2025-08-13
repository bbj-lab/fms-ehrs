#!/bin/bash

#SBATCH --job-name=proc-all
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --cpus-per-task=5
#SBATCH --mem=250GB
#SBATCH --time=2:00:00
#SBATCH --array=0-5

source preamble.sh

ni=2 nj=3
i=$((SLURM_ARRAY_TASK_ID % ni)) j=$((SLURM_ARRAY_TASK_ID / ni))

if ((SLURM_ARRAY_TASK_COUNT != ni * nj)); then
    echo "Warning:"
    echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"
    echo "ni*nj=$((ni * nj))"
fi

data_dirs=(
    /scratch/burkh4rt/data-mimic
    /scratch/burkh4rt/data-ucmc
)
splits=(
    train
    val
    test
)

python3 ../fms_ehrs/scripts/process_all_trajectories.py \
    --data_dir "${data_dirs[$i]}" \
    --data_version QC_day_stays_first_24h \
    --model_loc "${hm}/mdls-archive/llama1b-57928921-run1" \
    --splits "${splits[$j]}"

