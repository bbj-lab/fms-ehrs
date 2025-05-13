#!/bin/bash

export data_version=QC_noX_sigmas

j02=$(sbatch --parsable 02_tokenize_train_val_test_split.sh)

j03=$(
    sbatch --parsable \
        --depend=afterok:"${j02}" \
        03_tune_model.sh
)
