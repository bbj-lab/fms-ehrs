#!/bin/bash

export data_version=cerulean

j02=$(sbatch --parsable ../slurm/02_tokenize_train_val_test_split.sh)

j03=$(
    sbatch --parsable \
        --depend=afterok:"${j02}" \
        ../slurm/03_tune_model.sh
)
