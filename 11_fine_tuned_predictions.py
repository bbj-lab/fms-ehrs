#!/usr/bin/env python3

"""
fine-tune a pretrained model for sequence classification
"""

import os
import pathlib

import datasets as ds
import fire as fi
import numpy as np
import sklearn.metrics as skl_mets
import torch as t
from transformers import AutoModelForSequenceClassification, Trainer

from logger import get_logger
from util import rt_padding_to_left
from vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    model_dir: os.PathLike = "../clif-mdls-archive/mdl-llama1b-sft-57451707-clsfr",
    data_dir: os.PathLike = "../clif-data/day_stays_qc_first_24h-tokenized",
):

    model_dir, data_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (model_dir, data_dir),
    )

    # load and prep data
    splits = ("train", "val", "test")
    data_dirs = {s: data_dir.joinpath(s) for s in splits}

    vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

    dataset = (
        ds.load_dataset(
            "parquet",
            data_files={
                s: str(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
                for s in ("test",)
            },
            columns=["padded", "same_admission_death"],
        )
        .with_format("torch")
        .map(
            lambda x: {
                "input_ids": rt_padding_to_left(x["padded"], vocab("PAD")),
                "label": x["same_admission_death"],
            },
            remove_columns=["padded", "same_admission_death"],
        )
    )

    mort_true = dataset["test"]["label"].numpy()

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    trainer = Trainer(model=model)
    preds = trainer.predict(dataset["test"])
    logits = preds.predictions
    mort_probs = t.nn.functional.softmax(t.tensor(logits), dim=-1).numpy()[:, 1]
    mort_preds = np.argmax(mort_probs, axis=1)

    logger.info(
        "roc_auc: {:.3f}".format(
            skl_mets.roc_auc_score(y_true=mort_true, y_score=mort_probs)
        )
    )

    for met in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        logger.info(
            "{}: {:.3f}".format(
                met,
                getattr(skl_mets, f"{met}_score")(y_true=mort_true, y_pred=mort_preds),
            )
        )

    np.save(
        data_dirs["test"].joinpath(
            "sft-mortality-preds-{m}.npy".format(m=model_dir.stem)
        ),
        mort_probs,
    )


if __name__ == "__main__":
    fi.Fire(main)
