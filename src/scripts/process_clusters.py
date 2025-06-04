#!/usr/bin/env python3

"""
process clusters of 24h-representations
"""

import argparse
import pathlib
import pickle

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

from src.framework.logger import get_logger

pio.kaleido.scope.mathjax = None

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../clif-data-ucmc")
parser.add_argument("--data_version", type=str, default="QC_noX_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama1b-smol-59946181-hp-QC_noX",
)
parser.add_argument(
    "--dx_csv_loc",
    type=pathlib.Path,
    default="../../physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz",
)
parser.add_argument(
    "--icd_9_to_10_loc",
    type=pathlib.Path,
    default="../../data-leakage/data/icd_cm_9_to_10_mapping.csv.gz",
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc, dx_csv_loc, icd_9_to_10_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (
        args.data_dir_orig,
        args.data_dir_new,
        args.model_loc,
        args.dx_csv_loc,
        args.icd_9_to_10_loc,
        args.out_dir,
    ),
)

versions = ("orig", "new")
data_dirs = {
    v: (data_dir_orig if v == "orig" else data_dir_new).joinpath(
        f"{args.data_version}-tokenized", "test"
    )
    for v in versions
}

features = {
    v: np.load(data_dirs[v].joinpath("features-{m}.npy".format(m=model_loc.stem)))
    for v in versions
}

clusterers = dict()
labels = dict()

for v in versions:
    logger.info(f"{v}...")
    with open(
        data_dirs[v].joinpath("dbscan-reps-{m}-m100.pkl".format(m=model_loc.stem)),
        "rb",
    ) as fp:
        clusterers[v] = pickle.load(fp)

outcomes = ("same_admission_death", "long_length_of_stay", "icu_admission", "imv_event")
aux = {
    v: (
        pl.scan_parquet(
            data_dirs[v].joinpath(
                "tokens_timelines_outcomes.parquet",
            )
        )
        .select("hospitalization_id", *outcomes)
        .collect()
        .with_columns(hdbscan_label=pl.Series(clusterers[v].labels_))
    )
    for v in versions
}


mapping = pl.read_csv(icd_9_to_10_loc, infer_schema=False)
icd9_to_10 = dict(
    zip(
        mapping.select("icd_9").to_series().to_list(),
        mapping.select("icd_10").to_series().to_list(),
    )
)

icd_info = pl.read_csv(dx_csv_loc).with_columns(
    pl.when(pl.col("icd_version") == 9)
    .then(pl.col("icd_code").replace_strict(icd9_to_10, default="UNK"))
    .otherwise(pl.col("icd_code"))
    .alias("icd_code")
)

top_codes = (
    icd_info.select("icd_code")
    .to_series()
    .value_counts()
    .sort("count")
    .tail(250)
    .select("icd_code")
    .to_series()
    .to_list()
)[::-1]
top_codes.remove("UNK")

pivoted = (
    icd_info.filter(pl.col("icd_code").is_in(top_codes))
    .select("hadm_id", "icd_code")
    .pivot(index="hadm_id", on="icd_code", values="icd_code", aggregate_function="len")
    .fill_null(0)
    .select(pl.col("hadm_id").cast(str).alias("hospitalization_id"), *top_codes)
)

for v in versions:
    aux[v] = (
        aux[v]
        .join(pivoted, how="left", on="hospitalization_id", maintain_order="left")
        .fill_null(0)
    )

df_outcomes = (
    aux["orig"]
    .group_by("hdbscan_label")
    .agg([pl.col(c).mean() for c in outcomes[:2]])
    .sort("hdbscan_label")
    .to_pandas()
    .set_index("hdbscan_label")
)
fig = go.Figure(
    data=go.Heatmap(
        z=df_outcomes.values,
        x=list(df_outcomes.columns),
        y=list(df_outcomes.index),
        colorscale="Viridis",
        reversescale=False,
        showscale=True,
        zsmooth=False,
        xgap=1,
        ygap=1,
    )
)
fig.update_layout(
    title="Outcome rates by cluster (MIMIC)",
    xaxis=dict(title="outcomes", showgrid=False, zeroline=False),
    yaxis=dict(title="clusters", showgrid=False, zeroline=False, autorange="reversed"),
    height=900,
    width=300,
)
fig.write_image(
    out_dir.joinpath("dbscan-clusters-outcomes-{m}-m100.pdf".format(m=model_loc.stem))
)

df_dx = (
    aux["orig"]
    .group_by("hdbscan_label")
    .agg([pl.col(c).mean() for c in top_codes])
    .sort("hdbscan_label")
    .to_pandas()
    .set_index("hdbscan_label")
)
fig = go.Figure(
    data=go.Heatmap(
        z=df_dx.values,
        x=list(df_dx.columns),
        y=list(df_dx.index),
        colorscale="Viridis",
        reversescale=False,
        showscale=True,
        zsmooth=False,
        xgap=1,
        ygap=1,
    )
)
fig.update_layout(
    title="Diagnostic outcomes by cluster (MIMIC)",
    xaxis=dict(
        title="ICD codes",
        showgrid=False,
        zeroline=False,
        type="category",
        tickmode="linear",
    ),
    yaxis=dict(title="clusters", showgrid=False, zeroline=False, autorange="reversed"),
    height=900,
    width=3500,
)
fig.write_image(
    out_dir.joinpath("dbscan-clusters-dx-{m}-m100.pdf".format(m=model_loc.stem))
)


logger.info("---fin")
