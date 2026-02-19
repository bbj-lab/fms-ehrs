#!/usr/bin/env python3

"""
examine/process timeline completions (from just under 24h) from each test sequence
"""

import argparse
import gzip
import os
import pathlib
import pickle
import typing

import numpy as np
import polars as pl

from fms_ehrs.framework.logger import get_logger, log_classification_metrics
from fms_ehrs.framework.stats import bootstrap_ci
from fms_ehrs.framework.vocabulary import Vocabulary

Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument(
    "--data_version",
    type=str,
    default="W++_first_24h_llama-med-60358922_1-hp-W++_none_10pct_ppy",
)
parser.add_argument("--tto_version", type=str, default="W++_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument("--max_len", type=int, default=10_000)
parser.add_argument("--n_samp", type=int, default=20)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

df_test = pl.scan_parquet(
    data_dir / f"{args.data_version}-tokenized" / "test" / "tokens_timelines.parquet"
)

df_res = (
    df_test.join(
        pl.scan_parquet(
            data_dir
            / f"{args.tto_version}-tokenized"
            / "test"
            / "tokens_timelines_outcomes.parquet"
        ),
        how="left",
        on="hospitalization_id",
        validate="1:1",
    )
    .select("hospitalization_id", "same_admission_death")
    .collect()
)
outcome = df_res.select("same_admission_death").to_numpy().ravel()

vocab = Vocabulary().load(
    data_dir / f"{args.data_version}-tokenized" / "train" / "vocab.gzip"
)

with gzip.open(
    data_dir
    / f"{args.data_version}-tokenized"
    / "test"
    / f"gen_preds_ml{args.max_len}_nsamp{args.n_samp}.pkl.gz",
    "r+",
) as f:
    res = pickle.load(f)

check = np.array([[vocab("TL_END") in out for out in op] for op in res])
logger.info("{:.2f}% of timelines completed.".format(100 * check.mean()))
logger.info(
    "{:.2f}% of predictions completed.".format(100 * check.all(axis=1).mean(axis=0))
)

lens = np.array([[len(out) for out in op] for op in res])
logger.info("{:,} tokens generated.".format(lens.sum()))
logger.info("Avg len: {:.1f} (std: {:.1f})".format(lens.mean(), lens.std()))

preds = np.array([[vocab("DSCG_expired") in out for out in op] for op in res])


log_classification_metrics(y_true=outcome, y_score=preds.mean(axis=1), logger=logger)
logger.info(bootstrap_ci(y_true=outcome, y_score=preds.mean(axis=1)))

logger.info("Restricted to successful preds:")
log_classification_metrics(
    y_true=outcome[ck := check.all(axis=1)],
    y_score=preds.mean(axis=1)[ck],
    logger=logger,
)

logger.info(
    "Dropping unfinished runs -- restricted to persons with at least 1 finished run"
)
preds = np.nanmean(np.where(check, preds, np.nan), axis=1)

log_classification_metrics(
    y_true=outcome[np.isfinite(preds)], y_score=preds[np.isfinite(preds)], logger=logger
)


logger.info("---fin")
