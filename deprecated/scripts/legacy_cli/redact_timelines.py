#!/usr/bin/env python3

"""
Load timelines and model-determined importance, extract a subset (ICU stays),
and pare them down by removing k=5 events from the timelines determined
via information or randomly
"""

import argparse
import pathlib

import numpy as np
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.util import redact_tokens_times
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument(
    "--model_loc", type=pathlib.Path, default="../../mdls-archive/llama1b-57928921-run1"
)
parser.add_argument(
    "--method", choices=["top", "bottom", "random", "none", None], default=None
)
parser.add_argument(
    "--aggregation", choices=["max", "sum", "perplexity"], default="sum"
)
parser.add_argument("--k", type=int, default=None)
parser.add_argument("--pct", type=float, default=None)
parser.add_argument("--new_version", type=str, default="icu24h_top5-921_first_24h")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
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

    if args.method is not None and args.method != "none":
        tkn_new, tms_new = redact_tokens_times(
            tks_arr=tkn_icu,
            tms_arr=tms_icu,
            inf_arr=inf_icu,
            k=args.k,
            pct=args.pct,
            method=args.method,
            aggregation=args.aggregation,
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

    set_perms(df.write_parquet)(d_out.joinpath("tokens_timelines_outcomes.parquet"))
    df.drop(outcome_columns, strict=False).write_parquet(
        d_out.joinpath("tokens_timelines.parquet")
    )
    if s == "train":
        vocab.save(d_out.joinpath("vocab.gzip"))


logger.info("---fin")
