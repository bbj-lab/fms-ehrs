#!/bin/bash

#SBATCH --job-name=eval-ft-mdl
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-23

source preamble.sh

ni=2
nj=3
nk=4
i=$((SLURM_ARRAY_TASK_ID % ni))
jk=$((SLURM_ARRAY_TASK_ID / ni))
j=$((jk % nj))
k=$((jk / nj))

if ((SLURM_ARRAY_TASK_COUNT != ni * nj * nk)); then
    echo "Warning:"
    echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"
    echo "ni*nj*nk=$((ni * nj * nk))"
fi

data_dirs=("${hm}/clif-data" "${hm}/clif-data-ucmc")
models=(
    mdl-llama-orig-58789721-58997654-clsfr-same_admission_death
    mdl-llama-orig-58789721-58997873-clsfr-long_length_of_stay
    mdl-llama-orig-58789721-58998670-clsfr-imv_event
)
versions=(
    icu24h
    icu24h_top5-921
    icu24h_bot5-921
    icu24h_rnd5-921
)

python3 ../fms_ehrs/scripts/fine_tuned_predictions.py \
    --data_dir "${data_dirs[$i]}" \
    --data_version "${versions[$k]}_first_24h" \
    --model_loc "${hm}/clif-mdls-archive/${models[$j]}" \
    --outcome "${models[$j]##*-}"
