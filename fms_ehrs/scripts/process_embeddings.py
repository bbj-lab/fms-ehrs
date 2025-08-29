#!/usr/bin/env python3

"""
process embedded 24h-representations
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--name_orig", type=str, default="MIMIC")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../data-ucmc")
parser.add_argument("--name_new", type=str, default="UCMC")
parser.add_argument("--data_version", type=str, default="QC_noX_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama1b-original-59946215-hp-QC_noX",
)
parser.add_argument("--mapper", choices=["isomap", "umap", "pacmap"], default="pacmap")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

outcomes = ("same_admission_death", "long_length_of_stay", "icu_admission", "imv_event")

data_dir_orig, data_dir_new, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc, args.out_dir),
)

versions = ("orig", "new")
data_dirs = {
    v: (data_dir_orig if v == "orig" else data_dir_new).joinpath(
        f"{args.data_version}-tokenized", "test"
    )
    for v in versions
}

embd = {
    v: np.load(
        data_dirs[v].joinpath(
            "features-{typ}-{m}.npy".format(typ=args.mapper, m=model_loc.stem)
        )
    )
    for v in versions
}

flags = {
    v: (
        pl.scan_parquet(data_dirs[v].joinpath("tokens_timelines_outcomes.parquet"))
        .with_columns(
            [
                pl.when(pl.col(outcome))
                .then(pl.lit(outcome))
                .otherwise(pl.lit("n/a"))
                .alias(outcome)
                for outcome in outcomes
            ]
        )
        # .with_columns(flags=pl.concat_str(outcomes, separator=", ", ignore_nulls=True))
        .select(outcomes)
        .collect()
        .to_numpy()
        .transpose()
    )
    for v in versions
}

df = pd.concat(
    [
        pd.DataFrame(data=embd[v], columns=["dim1", "dim2"])
        .assign(version=args.name_orig if v == "orig" else args.name_new)
        .assign(**dict(zip(outcomes, flags[v])))
        for v in versions
    ],
    axis=0,
)

fig = px.scatter(df, x="dim1", y="dim2", color="version")
fig.update_layout(
    title="{typ} embedding of 24hr representations".format(typ=args.mapper),
    template="plotly_white",
)
fig.update_traces(marker=dict(size=1))
set_perms(fig.write_image)(
    out_dir.joinpath(
        "emb-{typ}-{m}-by-ds.pdf".format(typ=args.mapper, m=model_loc.stem)
    )
)

for out in outcomes[:2]:
    fig = px.scatter(df, x="dim1", y="dim2", color=out)
    fig.update_layout(
        title="{typ} embedding of 24hr representations".format(typ=args.mapper),
        template="plotly_white",
        font_family="CMU Serif, Times New Roman, serif",
    )
    fig.update_traces(marker=dict(size=1))
    set_perms(fig.write_image)(
        out_dir.joinpath(
            "emb-{typ}-{m}-{out}.pdf".format(typ=args.mapper, m=model_loc.stem, out=out)
        )
    )


logger.info("---fin")
