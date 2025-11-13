#!/bin/bash

# reports resource usage
sstat --jobs="${SLURM_JOB_ID}" -a --format=jobid,avecpu,maxrss
