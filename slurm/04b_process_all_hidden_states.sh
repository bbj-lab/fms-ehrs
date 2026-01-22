#!/bin/bash

#SBATCH --job-name=proc-all
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --cpus-per-task=5
#SBATCH --mem=500GB
#SBATCH --time=24:00:00
#SBATCH --array=0-5
##SBATCH --dependency=afterok:4887053,4887863

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
out_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
)
splits=(
    train
    val
    test
)

python3 ../fms_ehrs/scripts/process_all_trajectories.py \
    --data_dir "${data_dirs[$i]}" \
    --out_dir "${out_dirs[$i]}" \
    --data_version V21 \
    --model_loc "${hm}/mdls-archive/llama-med-4476655-hp-V21" \
    --splits "${splits[$j]}" \
    --all_layers

source postscript.sh
