#!/usr/bin/env python3

"""
Load timelines and a metric on tokens, and redact tokens from the events
according to the metric and method;
save results as a new data version
"""

import argparse
import gzip
import pathlib
import shutil

import numpy as np
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="Y21_icu24_first_24h")
parser.add_argument(
    "--model_loc", type=pathlib.Path, default="../../mdls-archive/gemma-5635921-Y21"
)
parser.add_argument(
    "--method",
    choices=["top", "bottom", "random", "none", "top_abs", "btm_abs"],
    default="none",
)
parser.add_argument("--metric", default="information")
parser.add_argument("--x_infm", action="store_true")
parser.add_argument("--pct", type=int, default=10)
parser.add_argument("--prefix_len", type=int, default=6)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")


rng = np.random.default_rng(seed=42)

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

vocab = Vocabulary().load(
    data_dir / f"{args.data_version}-tokenized" / "train" / "vocab.gzip"
)
pad_tkn = vocab("PAD")

new_version = (
    args.data_version.split("_first_24h")[0]
    + f"_red_{args.metric}"
    + ("-x-infm" if args.x_infm else "")
    + f"_{args.method}{args.pct}pct-{model_loc.stem}"
    + ("_first_24h" if args.data_version.endswith("_first_24h") else "")
)

splits = ("train", "val", "test")
for s in splits:
    d_in = data_dir / f"{args.data_version}-tokenized" / s
    d_out = data_dir / f"{new_version}-tokenized" / s
    d_out.mkdir(exist_ok=True, parents=True)

    if s == "train":
        for f in ("vocab.gzip", "config.yaml"):
            if (f_in := d_in / f).exists():
                shutil.copy2(f_in, d_out / f)

    tto = pl.read_parquet(d_in / "tokens_timelines_outcomes.parquet")
    met = np.load(
        gzip.open(
            d_in / "{met}-{mdl}.npy.gz".format(met=args.metric, mdl=model_loc.stem),
            mode="rb",
        )
    )
    if args.x_infm:
        infm = np.load(
            gzip.open(
                d_in / "information-{mdl}.npy.gz".format(mdl=model_loc.stem), mode="rb"
            )
        )
        met *= infm

    tkn = tto.select("padded").to_series().to_numpy()
    tms = tto.select("times").to_series().to_numpy()
    tkn_all = tto.select("tokens").to_series().to_list()
    paded_len = len(tkn[0])
    n_redacted = []
    seq_lens = []

    for i in range(len(tto)):
        end = np.searchsorted(tkn[i] == pad_tkn, 1)
        elen = end - args.prefix_len
        n_to_drop = int(elen * args.pct / 100) if args.method != "none" else 0

        pre_tk = tkn[i][: args.prefix_len]
        pre_tm = tms[i][: args.prefix_len]

        evt_tk = tkn[i][args.prefix_len : end]
        evt_tm = tms[i][args.prefix_len : end]
        evt_mt = met[i][args.prefix_len : end]

        match args.method:
            case "top":
                to_drop = np.argsort(evt_mt)[::-1][:n_to_drop]
            case "bottom":
                to_drop = np.argsort(evt_mt)[:n_to_drop]
            case "random":
                to_drop = rng.choice(len(evt_mt), size=n_to_drop, replace=False)
            case "none":
                to_drop = np.array([], dtype=int)
            case "top_abs":
                to_drop = np.argsort(np.abs(evt_mt))[::-1][:n_to_drop]
            case "btm_abs":
                to_drop = np.argsort(np.abs(evt_mt))[:n_to_drop]

        tkn[i] = np.concatenate(
            [
                pre_tk,
                np.delete(evt_tk, to_drop),
                np.full(
                    shape=len(tkn[i]) - len(pre_tk) - len(evt_tk) + len(to_drop),
                    fill_value=pad_tkn,
                ),
            ]
        )
        tms[i] = np.concatenate([pre_tm, np.delete(evt_tm, to_drop)])
        tkn_all[i] = np.delete(tkn_all[i], to_drop + args.prefix_len)

        assert len(tkn[i]) == paded_len
        n_redacted.append(n_to_drop)
        seq_lens.append(end - n_to_drop)

    tto = tto.with_columns(
        padded=pl.Series(tkn.tolist()).cast(pl.List(pl.Int64)),
        times=pl.Series(tms.tolist()).cast(pl.List(pl.Datetime("ms"))),
        tokens=pl.Series(tkn_all),
        seq_len=pl.Series(seq_lens),
    )
    set_perms(tto.write_parquet)(d_out / "tokens_timelines_outcomes.parquet")

    set_perms(
        pl.read_parquet(d_in / "tokens_timelines.parquet")
        .with_columns(
            padded=pl.Series(tkn.tolist()).cast(pl.List(pl.Int64)),
            times=pl.Series(tms.tolist()).cast(pl.List(pl.Datetime("ms"))),
            tokens=pl.Series(tkn_all),
            seq_len=pl.Series(seq_lens),
        )
        .write_parquet
    )(d_out / "tokens_timelines.parquet")

    logger.info(
        "Created version {} split {} redacting {:.2f} tokens on average".format(
            new_version, s, np.mean(n_redacted)
        )
    )


logger.info("---fin")
