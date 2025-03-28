#!/usr/bin/env python3

"""
fine-tune a pretrained model for sequence classification
"""

import os
import pathlib
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

from logger import get_logger
from util import rt_padding_to_left
from vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    model_loc: os.PathLike = "../clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630",
    data_dir: os.PathLike = "../clif-data",
    data_version: str = "day_stays_qc_first_24h",
    out_dir: os.PathLike = "../clif-mdls",
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
        "same_admission_death", "long_length_of_stay"
    ] = "same_admission_death",
    unif_rand_trunc: bool = False,
):

    model_loc, data_dir, out_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (model_loc, data_dir, out_dir),
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
            columns=["padded", "same_admission_death"],
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

    model = AutoModelForSequenceClassification.from_pretrained(model_loc)

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
        gradient_accumulation_steps=gradient_accumulation_steps,  # simulate larger batch sizes
        learning_rate=learning_rate,  # 2e-4 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
        num_train_epochs=n_epochs,
        save_total_limit=2,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        load_best_model_at_end=True,
        eval_strategy="steps",
        save_strategy="best",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model,
        train_dataset=dataset["train"].shuffle(generator=np_rng),
        eval_dataset=dataset["val"],
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(
        str(
            output_dir.joinpath(
                "mdl-{m}-{j}-clsfr-{o}{u}".format(
                    m=model_loc.stem,
                    j=jid,
                    o=outcome,
                    u="-urt" if unif_rand_trunc else "",
                )
            )
        )
    )


if __name__ == "__main__":
    fi.Fire(main)
