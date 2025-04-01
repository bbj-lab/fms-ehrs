#!/bin/bash

#SBATCH --job-name=eval-ft-mdl
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-7

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

div=2
quo=$((SLURM_ARRAY_TASK_ID / div))
rem=$((SLURM_ARRAY_TASK_ID % div))

data_dirs=("${hm}/clif-data" "/scratch/$(whoami)/clif-data")
outcomes=(same_admission_death long_length_of_stay icu_admission imv_event)

python3 "${name}.py" \
    --data_dir "${data_dirs[$rem]}" \
    --data_version QC_day_stays_first_24h \
    --model_loc "${hm}/clif-mdls-archive/mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death" \
    --outcome "${outcomes[$quo]}"
