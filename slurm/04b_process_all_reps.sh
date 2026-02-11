#!/bin/bash

#SBATCH --job-name=proc-all
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=tier3q
#SBATCH --cpus-per-task=5
#SBATCH --mem=500GB
#SBATCH --time=24:00:00
#SBATCH --array=0-1

source preamble.sh

data_dirs=(
    /scratch/burkh4rt/data-mimic
    /scratch/burkh4rt/data-ucmc
)
out_dirs=(
    "${hm}/data-mimic"
    "${hm}/data-ucmc"
)

python3 ../fms_ehrs/scripts/process_all_hidden_states.py \
    --data_dir "${data_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --out_dir "${out_dirs[$SLURM_ARRAY_TASK_ID]}" \
    --data_version Y21_first_24h \
    --model_loc "${hm}/mdls-archive/gemma-5635921-Y21" \
    --proto_dir "${hm}/data-mimic"

source postscript.sh
