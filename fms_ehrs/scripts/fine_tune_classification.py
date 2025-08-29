#!/usr/bin/env python3

"""
fine-tune a pretrained model for sequence classification
"""

import os
import pathlib
import sys
import typing

import datasets as ds
import fire as fi
import numpy as np
import scipy as sp
import sklearn.metrics as skl_mets
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.util import rt_padding_to_left
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    model_loc: os.PathLike = None,
    data_dir: os.PathLike = None,
    data_version: str = "day_stays_first_24h",
    out_dir: os.PathLike = None,
    n_epochs: int = 5,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    jid: str = os.getenv("SLURM_JOB_ID", ""),
    wandb_project: str = "mimic-sft-clsfr",
    metric_for_best_model: str = "eval_auc",
    greater_is_better: bool = True,
    outcome: typing.Literal[
        "same_admission_death", "long_length_of_stay", "icu_admission", "imv_event"
    ] = "same_admission_death",
    unif_rand_trunc: bool = False,
    tune: bool = False,
    training_fraction: float = 1.0,
) -> pathlib.PurePath | None:
    model_loc, data_dir, out_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (model_loc, data_dir, out_dir)
    )

    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_NAME"] = "{m}-{j}".format(m=model_loc.stem, j=jid)

    output_dir = out_dir.joinpath("{m}-{j}".format(m=model_loc.stem, j=jid))
    output_dir.mkdir(exist_ok=True, parents=True)

    # load and prep data
    splits = ("train", "val")
    data_dirs = {s: data_dir.joinpath(f"{data_version}-tokenized", s) for s in splits}
    np_rng = np.random.default_rng(42)
    vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

    dataset = (
        ds.load_dataset(
            "parquet",
            data_files={
                s: str(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
                for s in splits
            },
            columns=["padded", outcome],
        )
        .with_format("torch")
        .map(
            lambda x: {
                "input_ids": rt_padding_to_left(
                    x["padded"], vocab("PAD"), unif_rand_trunc=unif_rand_trunc
                ),
                "label": x[outcome],
            },
            remove_columns=["padded", outcome],
        )
    )

    assert 0 <= training_fraction <= 1.0
    if training_fraction < 1.0 - sys.float_info.epsilon:
        tr = dataset["train"].shuffle(generator=np_rng)
        n_tr = int(len(tr) * training_fraction)
        dataset["train"] = dataset["train"].select(range(n_tr))

    def model_init(trial=None):
        return AutoModelForSequenceClassification.from_pretrained(model_loc)

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True),
            "gradient_accumulation_steps": trial.suggest_int(
                "gradient_accumulation_steps", 1, 3
            ),
        }

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        probs = sp.special.softmax(logits, axis=1)[:, 1]
        preds = np.argmax(logits, axis=1)
        prec, rec, f1, _ = skl_mets.precision_recall_fscore_support(
            y_true=labels, y_pred=preds, pos_label=1, average="binary"
        )
        auc = skl_mets.roc_auc_score(y_true=labels, y_score=probs)
        return {"prec": prec, "rec": rec, "f1": f1, "auc": auc}

    # train model
    training_args = TrainingArguments(
        report_to="wandb",
        run_name="{m}-{j}".format(m=model_loc.stem, j=jid),
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=n_epochs,
        save_total_limit=2,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="best",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model_init(),
        model_init=model_init if tune else None,
        train_dataset=dataset["train"].shuffle(generator=np_rng),
        eval_dataset=dataset["val"],
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
    )

    if tune:
        best_trial = trainer.hyperparameter_search(
            direction="minimize", backend="optuna", hp_space=optuna_hp_space, n_trials=5
        )

        if os.getenv("RANK", "0") == "0":
            best_ckpt = sorted(
                output_dir.joinpath(f"run-{best_trial.run_id}").glob("checkpoint-*")
            ).pop()
            best_mdl_loc = out_dir.joinpath(
                "{m}-{j}-hp".format(m=model_loc.stem, j=jid)
            )
            set_perms(
                AutoModelForSequenceClassification.from_pretrained(
                    best_ckpt
                ).save_pretrained
            )(best_mdl_loc)

            return best_mdl_loc

    else:
        trainer.train()
        set_perms(trainer.save_model)(
            str(
                best_mdl_loc := output_dir.joinpath(
                    "mdl-{m}-{j}-clsfr-{o}{u}".format(
                        m=model_loc.stem,
                        j=jid,
                        o=outcome,
                        u="-urt" if unif_rand_trunc else "",
                    )
                )
            )
        )
        return best_mdl_loc


if __name__ == "__main__":
    fi.Fire(main)
