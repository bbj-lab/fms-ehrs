#!/usr/bin/env python3

"""
for a list of models, collect predictions and compare performance
"""

import argparse
import gzip
import pathlib
import pickle

import numpy as np
import pandas as pd
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.tokenizer import Tokenizer21

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="Y21_icu24_first_24h")
parser.add_argument("--proto_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument(
    "--model_loc", type=pathlib.Path, default="../../mdls-archive/gemma-5635921-Y21"
)
parser.add_argument(
    "--outcomes",
    nargs="+",
    default=[
        "same_admission_death",
        "long_length_of_stay",
        # "ama_discharge",
        # "hospice_discharge",
    ],
)
parser.add_argument(
    "--metrics",
    type=str,
    nargs="*",
    default=[
        "rel-imp-long_length_of_stay",
        "rel-imp-same_admission_death",
        "abs-imp-long_length_of_stay",
        "abs-imp-same_admission_death",
        "rel-gmm-long_length_of_stay",
        "rel-gmm-same_admission_death",
        "abs-gmm-long_length_of_stay",
        "abs-gmm-same_admission_death",
        # "importance-h2o-mean",
        # "importance-h2o-mean_log",
        # "importance-h2o-va-mean",
        # "importance-h2o-va-mean_log",
        # "importance-scissorhands-10",
        # "importance-scissorhands-20",
        # "importance-scissorhands-va-10",
        # "importance-scissorhands-va-20",
        # "importance-rollout-mean",
        # "importance-rollout-mean_log",
        # "importance-h2o-normed-mean",
        # "importance-h2o-normed-mean_log",
        # "information",
    ],
)
parser.add_argument("--ignore_prefix", type=int, default=5)
parser.add_argument("--truncate_at", type=int, default=300)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir, proto_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.out_dir, args.proto_dir, args.model_loc),
)

# load and prep data
splits = ("train", "val", "test")
data_dirs = {s: data_dir / f"{args.data_version}-tokenized" / s for s in splits}


tt = pl.scan_parquet(data_dirs["train"] / "tokens_timelines.parquet")

tkzr = Tokenizer21(
    data_dir=data_dirs["train"],
    vocab_path=data_dirs["train"] / "vocab.gzip",
    config_file=data_dirs["train"] / "config.yaml",
)


mets = {
    met: np.load(
        gzip.open(
            data_dirs["test"]
            / "{met}-{mdl}.npy.gz".format(met=met, mdl=model_loc.stem),
            "rb",
        )
    )
    for met in args.metrics
}
if args.ignore_prefix > 0:
    for k in mets.keys():
        mets[k][:, : args.ignore_prefix] = 0
if args.truncate_at > 0:
    for k in mets.keys():
        mets[k][:, args.truncate_at :] = 0

with open(
    proto_dir
    / (args.data_version + "-tokenized")
    / "train"
    / ("lda-gmm-protos-" + model_loc.stem + ".pkl"),
    "rb",
) as fp:
    pkl = pickle.load(fp)
    scaler = pkl["scaler"]
    models = pkl["models"]

reps = np.load(data_dirs["test"] / "features-{m}.npy".format(m=model_loc.stem))
tto = pl.read_parquet(
    data_dirs["test"] / "tokens_timelines_outcomes.parquet"
).with_columns(
    **{
        f"wt_md_{outcome}": models[outcome].predict_proba(scaler.transform(reps))[
            :, np.argmax(models[outcome].weights_)
        ]
        for outcome in args.outcomes
    },
    **{
        metric.replace("-", "_") + "q99": (
            (x := mets[metric]) > np.quantile(x[x > 0], 0.99)
        ).sum(axis=1)
        for metric in args.metrics
    },
)

for outcome in args.outcomes:
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, fmt_str_lengths=100):
        print(outcome)
        print(
            top10 := tto.filter(outcome).filter(pl.col("wt_md_" + outcome) > 0.9)
            .sort("abs_gmm_" + outcome + "q99", descending=True)
            .head(10)
        )
        print(top10.select("hospitalization_id").to_series().to_list())

# with pl.Config(tbl_rows=-1, tbl_width_chars=200, fmt_str_lengths=100):
#     print(
#         tt.select(
#             pl.col("tokens")
#             .explode()
#             .replace_strict(tkzr.vocab.reverse, return_dtype=pl.String)
#         )
#         .with_columns(
#             pairs=pl.concat_str(
#                 [pl.col("tokens"), pl.col("tokens").shift(-1)], separator=","
#             )
#         )
#         .filter(pl.col("tokens") != "TL_END")
#         .group_by("pairs")
#         .len()
#         .sort("len", descending=True)
#         .head(100)
#         .collect()
#     )
