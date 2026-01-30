#!/bin/bash

# sources standard scripts, exports paths

source ~/.bashrc 2> /dev/null

if [ -v SLURM_ARRAY_JOB_ID ]; then
    echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
fi

# `whoami` sometimes fails on cri*
whoami_str="$(command whoami 2> /dev/null || echo burkh4rt)"

case "$(uname -n)" in
    cri*)
        hm="/gpfs/data/bbj-lab/users/${whoami_str}"
        HF_HOME=/gpfs/data/bbj-lab/cache/huggingface/
        WANDB_CACHE_DIR="/scratch/${whoami_str}/"
        WANDB_DIR="/scratch/${whoami_str}/"
        name=$(scontrol show job "$SLURM_JOBID" \
            | grep -m 1 "Command=" \
            | cut -d "=" -f2 \
            | xargs -I {} basename {} .sh)
        jname=$(scontrol show job "$SLURM_JOBID" \
            | grep -oP 'JobName=\K\S+')
        ;;
    bbj-lab*)
        hm="/home/${whoami_str}"
        HF_HOME="/home/${whoami_str}/cache/huggingface/"
        name="adhoc"
        ;;
    *)
        echo "Unknown host $(uname -n)"
        exit 1
        ;;
esac

parent_dir="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
source "${parent_dir}/.venv/bin/activate" 2> /dev/null
PYTHONPATH="${parent_dir}:$PYTHONPATH"

export hm name parent_dir HF_HOME WANDB_CACHE_DIR WANDB_DIR PYTHONPATH
