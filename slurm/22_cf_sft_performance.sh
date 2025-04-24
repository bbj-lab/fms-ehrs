#!/bin/bash

#SBATCH --job-name=cf-mdls
#SBATCH --output=./output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=3:00:00
##SBATCH --dependency=afterok:59000155_[0-6]
#SBATCH --array=0-7

source preamble.sh

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

div=2
quo=$((SLURM_ARRAY_TASK_ID / div))
rem=$((SLURM_ARRAY_TASK_ID % div))

data_dirs=("${hm}/clif-data" "${hm}/clif-data-ucmc")

case ${quo} in
    0)
        outcome=same_admission_death
        models=(
            "${hm}/clif-mdls-archive/mdl-llama-orig-58789721-58997654-clsfr-same_admission_death"
            "${hm}/clif-mdls-archive/mdl-llama-large-58788825-58998910-clsfr-same_admission_death"
            "${hm}/clif-mdls-archive/mdl-llama-med-58788824-59002326-clsfr-same_admission_death"
            "${hm}/clif-mdls-archive/mdl-llama-small-58741567-59019015-clsfr-same_admission_death"
            "${hm}/clif-mdls-archive/mdl-llama-smol-58761427-59019126-clsfr-same_admission_death"
            "${hm}/clif-mdls-archive/mdl-llama-tiny-58761428-59019258-clsfr-same_admission_death"
            "${hm}/clif-mdls-archive/mdl-llama-teensy-58741565-59019371-clsfr-same_admission_death"
            "${hm}/clif-mdls-archive/mdl-llama-wee-58996725-59020536-clsfr-same_admission_death"
            "${hm}/clif-mdls-archive/mdl-llama-bitsy-58996726-59025602-clsfr-same_admission_death"
            "${hm}/clif-mdls-archive/mdl-llama-micro-58996720-59026365-clsfr-same_admission_death"
        )
        ;;
    1)
        outcome=long_length_of_stay
        models=(
            "${hm}/clif-mdls-archive/mdl-llama-orig-58789721-58997873-clsfr-long_length_of_stay"
            "${hm}/clif-mdls-archive/mdl-llama-large-58788825-58999175-clsfr-long_length_of_stay"
            "${hm}/clif-mdls-archive/mdl-llama-med-58788824-59013037-clsfr-long_length_of_stay"
            "${hm}/clif-mdls-archive/mdl-llama-small-58741567-59019016-clsfr-long_length_of_stay"
            "${hm}/clif-mdls-archive/mdl-llama-smol-58761427-59019145-clsfr-long_length_of_stay"
            "${hm}/clif-mdls-archive/mdl-llama-tiny-58761428-59019266-clsfr-long_length_of_stay"
            "${hm}/clif-mdls-archive/mdl-llama-teensy-58741565-59019388-clsfr-long_length_of_stay"
            "${hm}/clif-mdls-archive/mdl-llama-wee-58996725-59025304-clsfr-long_length_of_stay"
            "${hm}/clif-mdls-archive/mdl-llama-bitsy-58996726-59026260-clsfr-long_length_of_stay"
            "${hm}/clif-mdls-archive/mdl-llama-micro-58996720-59026481-clsfr-long_length_of_stay"
        )
        ;;
    2)
        outcome=icu_admission
        models=(
            "${hm}/clif-mdls-archive/mdl-llama-orig-58789721-58997981-clsfr-icu_admission"
            "${hm}/clif-mdls-archive/mdl-llama-large-58788825-58999593-clsfr-icu_admission"
            "${hm}/clif-mdls-archive/mdl-llama-med-58788824-59018866-clsfr-icu_admission"
            "${hm}/clif-mdls-archive/mdl-llama-small-58741567-59019033-clsfr-icu_admission"
            "${hm}/clif-mdls-archive/mdl-llama-smol-58761427-59019233-clsfr-icu_admission"
            "${hm}/clif-mdls-archive/mdl-llama-tiny-58761428-59019362-clsfr-icu_admission"
            "${hm}/clif-mdls-archive/mdl-llama-teensy-58741565-59019402-clsfr-icu_admission"
            "${hm}/clif-mdls-archive/mdl-llama-wee-58996725-59025432-clsfr-icu_admission"
            "${hm}/clif-mdls-archive/mdl-llama-bitsy-58996726-59026270-clsfr-icu_admission"
            "${hm}/clif-mdls-archive/mdl-llama-micro-58996720-59026499-clsfr-icu_admission"
        )
        ;;
    3)
        outcome=imv_event
        models=(
            "${hm}/clif-mdls-archive/mdl-llama-orig-58789721-58998670-clsfr-imv_event"
            "${hm}/clif-mdls-archive/mdl-llama-large-58788825-59000154-clsfr-imv_event"
            "${hm}/clif-mdls-archive/mdl-llama-med-58788824-59018948-clsfr-imv_event"
            "${hm}/clif-mdls-archive/mdl-llama-small-58741567-59019122-clsfr-imv_event"
            "${hm}/clif-mdls-archive/mdl-llama-smol-58761427-59019254-clsfr-imv_event"
            "${hm}/clif-mdls-archive/mdl-llama-tiny-58761428-59019363-clsfr-imv_event"
            "${hm}/clif-mdls-archive/mdl-llama-teensy-58741565-58996783-clsfr-imv_event"
            "${hm}/clif-mdls-archive/mdl-llama-wee-58996725-59025505-clsfr-imv_event"
            "${hm}/clif-mdls-archive/mdl-llama-bitsy-58996726-59026353-clsfr-imv_event"
            "${hm}/clif-mdls-archive/mdl-llama-micro-58996720-59000229-clsfr-imv_event"
        )
        ;;
    *)
        echo "Invalid quo: ${quo}"
        ;;
esac

python3 ../src/scripts/aggregate_sft_preds.py \
    --data_dir "${data_dirs[$rem]}" \
    --out_dir "${hm}/figs" \
    --data_version QC_day_stays_first_24h \
    --outcome "$outcome" \
    --models "${models[@]}"
