#!/usr/bin/env python3

"""
make predictions using a fine-tuned model for sequence classification
"""

import os
import pathlib
import typing

import datasets as ds
import fire as fi
import numpy as np
import torch as t
from transformers import AutoModelForSequenceClassification, Trainer

from fms_ehrs.framework.logger import get_logger, log_classification_metrics
from fms_ehrs.framework.storage import fix_perms
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
    outcome: typing.Literal[
        "same_admission_death", "long_length_of_stay", "icu_admission", "imv_event"
    ] = "same_admission_death",
):
    model_loc, data_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (model_loc, data_dir)
    )

    # load and prep data
    splits = ("train", "val", "test")
    data_dirs = {s: data_dir.joinpath(f"{data_version}-tokenized", s) for s in splits}

    vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

    dataset = (
        ds.load_dataset(
            "parquet",
            data_files={
                s: str(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
                for s in ("test",)
            },
        )
        .map(
            lambda x: {
                "same_admission_death_24h": False,
                "long_length_of_stay_24h": False,
            }
        )
        .select_columns(["padded", outcome, f"{outcome}_24h"])
        .with_format("torch")
        .map(
            lambda x: {
                "input_ids": rt_padding_to_left(x["padded"], vocab("PAD")),
                "label": x[outcome],
            },
            remove_columns=["padded", outcome],
        )
    )

    y_true = dataset["test"]["label"].numpy()
    qualifier = ~dataset["test"][
        f"{outcome}_24h"
    ].numpy()  # qualifies if event did not occur in first 24h

    model = AutoModelForSequenceClassification.from_pretrained(model_loc)
    trainer = Trainer(model=model)
    preds = trainer.predict(dataset["test"])
    logits = preds.predictions
    y_score = t.nn.functional.softmax(t.tensor(logits), dim=-1).numpy()[:, 1]

    fix_perms(np.save)(
        data_dirs["test"].joinpath(
            "sft-{o}-preds-{m}.npy".format(o=outcome, m=model_loc.stem)
        ),
        np.column_stack((y_score, qualifier)),
    )

    log_classification_metrics(
        y_true=y_true[qualifier], y_score=y_score[qualifier], logger=logger
    )


if __name__ == "__main__":
    fi.Fire(main)
