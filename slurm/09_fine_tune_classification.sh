#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --array=2-3

source preamble.sh

case "${SLURM_ARRAY_TASK_ID}" in
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

wandb_project+="-urt"

torchrun --nproc_per_node=8 \
    ../src/scripts/fine_tune_classification.py \
    --model_loc "${hm}/clif-mdls-archive/llama1b-57928921-run1" \
    --data_dir "${hm}/clif-data" \
    --data_version QC_day_stays_first_24h \
    --out_dir "${hm}/clif-mdls" \
    --n_epochs 10 \
    --learning_rate 0.00002 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --outcome "$outcome" \
    --wandb_project "$wandb_project" \
    --unif_rand_trunc True
