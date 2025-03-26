#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1

source preamble.sh
urt=true

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        outcome=same_admission_death
        wandb_project=mimic-sft-clsfr-mort
        ;;
    1)
        outcome=long_length_of_stay
        wandb_project=mimic-sft-clsfr-llos
        ;;
    *)
        echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
        ;;
esac

if ((urt)); then
    wandb_project+="-urt"
fi

torchrun --nproc_per_node=8 \
    "${name}.py" \
    --model_loc "${hm}/clif-mdls-archive/mdl-QC_day_stays-llama1b-57895023" \
    --data_dir "${hm}/clif-data" \
    --data_version QC_day_stays_first_24h \
    --out_dir "${hm}/clif-mdls" \
    --n_epochs 5 \
    --learning_rate 0.00002 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --outcome "$outcome" \
    --wandb_project "$wandb_project" \
    --unif_rand_trunc "$urt"
