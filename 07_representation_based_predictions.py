#!/usr/bin/env python3

"""
make some simple predictions outcomes ~ features
break down performance by ICU admission type
"""

import os
import pathlib

import fire as fi
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import sklearn.metrics as skl_mets

pd.options.display.float_format = "{:,.3f}".format
pd.options.display.max_columns = None
pd.options.display.width = 250
pd.options.display.max_colwidth = 100

from logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    data_dir: os.PathLike = "../clif-data",
    data_version: str = "day_stays_qc_first_24h",
    model_loc: os.PathLike = "../clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630",
    fast: bool = True,
):

    data_dir, model_loc = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (data_dir, model_loc),
    )

    splits = ("train", "val", "test")
    data_dirs = dict()
    for s in splits:
        data_dirs[s] = data_dir.joinpath(f"{data_version}-tokenized", s)

    """ classifier for admission mortality
    """
    logger.info("admission mortality".upper().ljust(79, "-"))

    data = dict()
    for s in ("train", "val"):
        feats = np.load(
            data_dirs[s].joinpath("features-{m}.npy".format(m=model_loc.stem))
        )
        mort = (
            pl.scan_parquet(data_dirs[s].joinpath("outcomes.parquet"))
            .select("same_admission_death")
            .collect()
            .to_numpy()
            .ravel()
        )
        data[s] = lgb.Dataset(feats, label=mort)

    bst = lgb.train(
        {"metric": "auc", "objective": "binary", "force_col_wise": True}
        | ({} if fast else {"learning_rate": 0.05}),
        data["train"],
        10 if fast else 1000,
        valid_sets=[data["val"]],
    )
    mort_pred = bst.predict(
        np.load(data_dirs["test"].joinpath("features-{m}.npy".format(m=model_loc.stem)))
    )
    mort_true = (
        pl.scan_parquet(data_dirs["test"].joinpath("outcomes.parquet"))
        .select("same_admission_death")
        .cast(pl.Int64)
        .collect()
        .to_numpy()
    )

    logger.info(
        "roc_auc: {:.3f}".format(
            skl_mets.roc_auc_score(y_true=mort_true, y_score=mort_pred)
        )
    )

    for met in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
    ):
        logger.info(
            "{}: {:.3f}".format(
                met,
                getattr(skl_mets, f"{met}_score")(
                    y_true=mort_true, y_pred=np.round(mort_pred)
                ),
            )
        )

    """ classifier for long length of stay
    """
    logger.info("long length of stay".upper().ljust(79, "-"))

    data = dict()
    for s in ("train", "val"):
        feats = np.load(
            data_dirs[s].joinpath("features-{m}.npy".format(m=model_loc.stem))
        )
        llos = (
            pl.scan_parquet(data_dirs[s].joinpath("outcomes.parquet"))
            .select("long_length_of_stay")
            .collect()
            .to_numpy()
            .astype(int)
            .ravel()
        )
        data[s] = lgb.Dataset(feats, label=llos)

    bst = lgb.train(
        {"metric": "auc", "objective": "binary", "force_col_wise": True}
        | ({} if fast else {"learning_rate": 0.05}),
        data["train"],
        10 if fast else 1000,
        valid_sets=[data["val"]],
    )
    llos_pred = bst.predict(
        np.load(data_dirs["test"].joinpath("features-{m}.npy".format(m=model_loc.stem)))
    )
    llos_true = (
        pl.scan_parquet(data_dirs["test"].joinpath("outcomes.parquet"))
        .select("long_length_of_stay")
        .cast(pl.Int64)
        .collect()
        .to_numpy()
    )

    logger.info(
        "roc_auc: {:.3f}".format(
            skl_mets.roc_auc_score(y_true=llos_true, y_score=llos_pred)
        )
    )

    for met in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
    ):
        logger.info(
            "{}: {:.3f}".format(
                met,
                getattr(skl_mets, f"{met}_score")(
                    y_true=llos_true, y_pred=np.round(llos_pred)
                ),
            )
        )

    np.save(
        data_dirs["test"].joinpath(
            "feat-based-mort-llos-preds-{m}.npy".format(m=model_loc.stem)
        ),
        np.column_stack([mort_pred, llos_pred]),
    )


if __name__ == "__main__":
    fi.Fire(main)
