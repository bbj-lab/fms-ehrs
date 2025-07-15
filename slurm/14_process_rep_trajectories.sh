#!/bin/bash

#SBATCH --job-name=process-jumps
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --cpus-per-task=5
#SBATCH --mem=250GB
#SBATCH --time=30:00
#SBATCH --array=0-1

source preamble.sh

model=llama-med-60358922_1-hp-W++
data_dirs=("${hm}/clif-data" "${hm}/clif-data-ucmc")

python3 ../fms_ehrs/scripts/process_representation_trajectories.py \
    --data_dir "${data_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --data_version "${model##*-}" \
    --model_loc "${hm}/clif-mdls-archive/${model}" \
    --save_jumps True
