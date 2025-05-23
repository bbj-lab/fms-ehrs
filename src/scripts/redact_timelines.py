#!/usr/bin/env python3

"""
Load timelines and model-determined importance, extract a subset (ICU stays),
and pare them down by removing k=5 events from the timelines determined
via information or randomly
"""

import argparse
import copy
import pathlib
import typing

import numpy as np
import polars as pl

from src.framework.logger import get_logger
from src.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama1b-57928921-run1",
)
parser.add_argument(
    "--method",
    choices=["top_k", "bottom_k", "random_k", "none"],
    default="top_k",
)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--new_version", type=str, default="icu24h_top5-921_first_24h")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

rng = np.random.default_rng(42)

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc),
)

outcome_columns = (
    "icu_admission_24h",
    "imv_event_24h",
    "length_of_stay",
    "same_admission_death",
    "long_length_of_stay",
    "icu_admission",
    "imv_event",
)


def redact_tokens_times(
    tks_arr: np.array,
    tms_arr: np.array,
    inf_arr: np.array,
    k: int = 5,
    method: typing.Literal["top_k", "bottom_k", "random_k"] = "top_k",
) -> tuple[np.array, np.array]:
    """given an array `tks_arr` of arrays of tokens and an array `tms_arr` of
    arrays of times, and an array `inf_arr` containing the information content
    up to a certain cutoff of the tokens in each timeline, iterate through the
    timelines and drop all tokens corresponding to times containing `k` (`top_k`)
    most informative, (`bottom_k`) least informative, or (`random_k`) randomly
    chosen tokens (not including the prefix, which we always keep)
    """
    assert len(tks_arr) == len(tms_arr) == len(inf_arr)
    tks_new = copy.deepcopy(tks_arr)
    tms_new = copy.deepcopy(tms_arr)
    for i in range(len(tks_new)):
        tks, tms = tks_arr[i], tms_arr[i]
        tlen = min(len(tks), len(tms))
        tks, tms = tks[:tlen], tms[:tlen]
        tms_unq, idx = np.unique(tms, return_inverse=True)
        if method in ("top_k", "bottom_k"):
            infm = inf_arr[i, :tlen]
            result = np.full(tms_unq.shape, -np.inf)
            np.maximum.at(result, idx, infm)
            srt = np.argsort(result)
            if method == "top_k":
                srt = srt[::-1]
        elif method == "random_k":
            srt = rng.permutation(len(tms_unq))
        else:
            raise Exception(f"Check {method=}")
        srt = srt[srt != idx[0]]  # don't drop prefix
        to_drop = srt[:k]
        tks_new[i] = tks[~np.isin(idx, to_drop)]
        tms_new[i] = tms[~np.isin(idx, to_drop)]
    return tks_new, tms_new


vocab = Vocabulary().load(
    data_dir.joinpath(f"{args.data_version}-tokenized", "train", "vocab.gzip")
)
pad_tkn = vocab("PAD")

splits = ("train", "val", "test")
for s in splits:
    dv = data_dir.joinpath(f"{args.data_version}-tokenized", s)
    d_out = data_dir.joinpath(f"{args.new_version}-tokenized", s)
    d_out.mkdir(exist_ok=True, parents=True)

    df = pl.read_parquet(dv.joinpath("tokens_timelines_outcomes.parquet"))
    infm = np.load(dv.joinpath("log_probs-{m}.npy".format(m=model_loc.stem))) / -np.log(
        2
    )

    icu_adm = df.select("icu_admission_24h").to_numpy().ravel()
    df_icu = df.filter("icu_admission_24h")

    tkn_icu = df_icu.select("padded").to_series().to_numpy()
    tms_icu = df_icu.select("times").to_series().to_numpy()
    inf_icu = infm[icu_adm]
    max_pad = len(tkn_icu[0])

    if args.method and args.method != "none":
        tkn_new, tms_new = redact_tokens_times(
            tks_arr=tkn_icu,
            tms_arr=tms_icu,
            inf_arr=inf_icu,
            k=args.k,
            method=args.method,
        )
        df = (
            df_icu.with_columns(
                padded=pl.Series(
                    [x.tolist() for x in tkn_new], dtype=pl.List(pl.Int64)
                ),
                times=pl.Series(
                    [x.tolist() for x in tms_new],
                    dtype=pl.List(pl.Datetime(time_unit="ms")),
                ),
            )
            .with_columns(redacted_len=pl.col("padded").list.len())
            .with_columns(
                padded=pl.concat_list(
                    "padded",
                    pl.lit(pad_tkn).repeat_by(max_pad - pl.col("redacted_len")),
                )
            )
        )
    else:
        df = df_icu

    df.write_parquet(d_out.joinpath("tokens_timelines_outcomes.parquet"))
    df.drop(outcome_columns, strict=False).write_parquet(
        d_out.joinpath("tokens_timelines.parquet")
    )
    if s == "train":
        vocab.save(d_out.joinpath("vocab.gzip"))

logger.info("---fin")
