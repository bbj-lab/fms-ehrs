#!/bin/bash

export versions=(
    icu24h
    icu24h_top5-921
    icu24h_bot5-921
    icu24h_rnd5-921
)

j05=$(
    sbatch --parsable \
        --chdir=../slurm \
        ../slurm/05_extract_hidden_states.sh
)

j07=$(
    sbatch --parsable \
        --depend=afterok:"${j05}" \
        --chdir=../slurm \
        ../slurm/07_find_outliers_oos.sh
)

j08=$(
    sbatch --parsable \
        --depend=afterok:"${j07}" \
        --chdir=../slurm \
        --array="0-$((${#versions[*]} - 1))" \
        ../slurm/08_transfer_rep_based_preds.sh
)

echo "$j05" "$j07" "$j08"
