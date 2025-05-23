#!/bin/bash

# sources standard scripts, exports paths

if [ -v SLURM_ARRAY_JOB_ID ]; then
    echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
fi

hm="/gpfs/data/bbj-lab/users/$(whoami)"
name=$(scontrol show job "$SLURM_JOBID" \
    | grep -m 1 "Command=" \
    | cut -d "=" -f2 \
    | xargs -I {} basename {} .sh)
parent_dir="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
export hm name parent_dir

source ~/.bashrc 2> /dev/null
source "${parent_dir}/venv/bin/activate" 2> /dev/null

HF_HOME=/gpfs/data/bbj-lab/cache/huggingface/
WANDB_CACHE_DIR="/scratch/$(whoami)/"
WANDB_DIR="/scratch/$(whoami)/"
PYTHONPATH="${parent_dir}:$PYTHONPATH"
export HF_HOME WANDB_CACHE_DIR WANDB_DIR PYTHONPATH
