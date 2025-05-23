#!/bin/bash

j02=$(
    sbatch --parsable \
        --chdir=../slurm \
        ../slurm/02_tokenize_train_val_test_split.sh
)

j03=$(
    sbatch --parsable \
        --depend=afterok:"${j02}" \
        --chdir=../slurm \
        ../slurm/03_tune_model.sh
)
