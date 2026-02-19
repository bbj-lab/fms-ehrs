#!/usr/bin/env python3

"""
select a subset of a dataset by filtering
"""

import argparse
import gzip
import pathlib
import shutil

import numpy as np
import polars as pl
from plotly import io as pio

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

try:
    pio.defaults.mathjax = None
except AttributeError:
    pio.kaleido.scope.mathjax = None

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--from_version", type=str, default="Y21_first_24h")
parser.add_argument("--to_version", type=str, default="Y21_icu24_first_24h")
parser.add_argument("--filter_col", type=str, default="icu_admission_24h")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

splits = ("train", "val", "test")
data_dir = pathlib.Path(args.data_dir).expanduser().resolve()

data_dirs_from = {s: data_dir / f"{args.from_version}-tokenized" / s for s in splits}
data_dirs_to = {s: data_dir / f"{args.to_version}-tokenized" / s for s in splits}
for dd in data_dirs_to.values():
    dd.mkdir(parents=True, exist_ok=True)

for s in splits:
    tto = pl.read_parquet(data_dirs_from[s] / "tokens_timelines_outcomes.parquet")
    mask = tto.select(args.filter_col).to_series().to_numpy().flatten()
    tto.filter(args.filter_col).write_parquet(
        data_dirs_to[s] / "tokens_timelines_outcomes.parquet"
    )
    pl.read_parquet(data_dirs_from[s] / "tokens_timelines.parquet").filter(
        pl.Series(mask)
    ).write_parquet(data_dirs_to[s] / "tokens_timelines.parquet")

    for f in data_dirs_from[s].glob("*.npy.gz"):
        with gzip.open(f, "rb") as fin:
            arr = np.load(fin)
        arr = arr[mask]
        out_f = data_dirs_to[s] / f.name
        with gzip.open(out_f, "wb") as fout:
            np.save(fout, arr)
        set_perms(out_f)

    for f in data_dirs_from[s].glob("*.npy"):
        arr = np.load(f)
        arr = arr[mask]
        out_f = data_dirs_to[s] / f.name
        np.save(out_f, arr)
        set_perms(out_f)

for file in ("config.yaml", "vocab.gzip"):
    if (f := data_dirs_from["train"] / file).exists():
        shutil.copy2(f, data_dirs_to["train"])
        set_perms(data_dirs_to["train"] / file)
