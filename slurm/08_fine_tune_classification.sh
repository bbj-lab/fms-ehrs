#!/bin/bash

#SBATCH --job-name=sft-w++
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3

source preamble.sh

case ${SLURM_ARRAY_TASK_ID} in
    0)
        outcome=same_admission_death
        wandb_project=mimic-sft-clsfr-mort
        ;;
    1)
        outcome=long_length_of_stay
        wandb_project=mimic-sft-clsfr-llos
        ;;
    2)
        outcome=icu_admission
        wandb_project=mimic-sft-clsfr-icua
        ;;
    3)
        outcome=imv_event
        wandb_project=mimic-sft-clsfr-imve
        ;;
    *)
        echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
        ;;
esac

res=$(
    python3 ../fms_ehrs/scripts/fine_tune_classification.py \
        --model_loc "${hm}/mdls-archive/llama-med-4476655-hp-V21" \
        --data_dir "${hm}/data-mimic" \
        --data_version V21_first_24h \
        --out_dir "${hm}/mdls" \
        --n_epochs 10 \
        --learning_rate 0.00005 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --outcome "$outcome" \
        --tune True \
        --wandb_project "$wandb_project" \
        --jid "'${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
)

echo "$res"
