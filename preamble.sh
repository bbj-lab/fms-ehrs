#!/bin/bash

# sources standard scripts
# exports name `name` of slurm calling script
# and home `hm` directory

source ~/.bashrc 2> /dev/null
source venv/bin/activate 2> /dev/null

hm="/gpfs/data/bbj-lab/users/$(whoami)"
name=$(scontrol show job "$SLURM_JOBID" |
    grep -m 1 "Command=" |
    cut -d "=" -f2 |
    xargs -I {} basename {} .sh)
export hm name
