#!/bin/bash

# sources standard scripts
# exports paths

hm="/gpfs/data/bbj-lab/users/$(whoami)"
name=$(scontrol show job "$SLURM_JOBID" |
    grep -m 1 "Command=" |
    cut -d "=" -f2 |
    xargs -I {} basename {} .sh)
export hm name

source ~/.bashrc 2> /dev/null
source "$hm/clif-tokenizer/venv/bin/activate" 2> /dev/null

HF_HOME=/gpfs/data/bbj-lab/cache/huggingface/
WANDB_CACHE_DIR="/scratch/$(whoami)/"
WANDB_DIR="/scratch/$(whoami)/"
PYTHONPATH="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"):$PYTHONPATH"
export HF_HOME WANDB_CACHE_DIR WANDB_DIR PYTHONPATH
