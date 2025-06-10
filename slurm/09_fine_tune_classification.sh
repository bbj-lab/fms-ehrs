#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-39

source preamble.sh

div=4
quo=$((SLURM_ARRAY_TASK_ID / div))
rem=$((SLURM_ARRAY_TASK_ID % div))

source preamble.sh

models=(
    llama-orig-58789721
    llama-large-58788825
    llama-med-58788824
    llama-small-58741567
    llama-smol-58761427
    llama-tiny-58761428
    llama-teensy-58741565
    llama-wee-58996725
    llama-bitsy-58996726
    llama-micro-58996720
)

case ${rem} in
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
    torchrun --nproc_per_node=8 \
        ../fms_ehrs/scripts/fine_tune_classification.py \
        --model_loc "${hm}/clif-mdls-archive/${models[$quo]}" \
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
        --jid "'${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
)

echo "$res"
