#!/bin/bash

#SBATCH --job-name=tfr-sft
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11

source preamble.sh

case $((SLURM_ARRAY_TASK_ID / 3)) in
    0)
        outcome=same_admission_death
        wandb_project=ucmc-sft-clsfr-mort
        model=mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death
        ;;
    1)
        outcome=long_length_of_stay
        wandb_project=ucmc-sft-clsfr-llos
        model=mdl-llama1b-57928921-run1-58134628-clsfr-long_length_of_stay
        ;;
    2)
        outcome=icu_admission
        wandb_project=ucmc-sft-clsfr-icua
        model=mdl-llama1b-57928921-run1-58165534-clsfr-icu_admission
        ;;
    3)
        outcome=imv_event
        wandb_project=ucmc-sft-clsfr-imve
        model=mdl-llama1b-57928921-run1-58165531-clsfr-imv_event
        ;;
    *)
        echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" && exit
        ;;
esac

training_fractions=(0.1 0.2 0.5)
training_fraction=${training_fractions[$((SLURM_ARRAY_TASK_ID % 3))]}

res=$(
    torchrun --nproc_per_node=8 \
        ../src/scripts/fine_tune_classification.py \
        --model_loc "${hm}/clif-mdls-archive/${model}" \
        --data_dir "${hm}/clif-data-ucmc" \
        --data_version QC_day_stays_first_24h \
        --out_dir "${hm}/clif-mdls" \
        --n_epochs 10 \
        --learning_rate 0.00002 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --outcome "$outcome" \
        --wandb_project "$wandb_project" \
        --tune True \
        --training_fraction "$training_fraction" \
        --jid "'${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
)

echo "$res"
