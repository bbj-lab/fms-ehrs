#!/usr/bin/env python3

"""
- determine:
    - if outliers @24h correlate with outliers from full timeline
    - if outliers correlate with adverse events
"""

import os
import pathlib

import numpy as np
import plotly.express as px
import polars as pl

# import sklearn.manifold as skl_mfld
import sklearn.decomposition as skl_decomp
import sklearn.ensemble as skl_ens

from logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

data_dir: os.PathLike = "../clif-data"
data_version: str = "day_stays_qc_first_24h"
model_loc: os.PathLike = "../clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630"
out_dir: os.PathLike = "../"


data_dir, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (data_dir, model_loc, out_dir),
)

splits = ("train", "val", "test")
data_dirs = dict()
feats = dict()
mort = dict()
llos = dict()
for s in splits:
    data_dirs[s] = data_dir.joinpath(f"{data_version}-tokenized", s)
    feats[s] = np.load(
        data_dirs[s].joinpath("features-{m}.npy".format(m=model_loc.stem))
    )
    mort[s] = (
        pl.scan_parquet(data_dirs[s].joinpath("outcomes.parquet"))
        .select("same_admission_death")
        .collect()
        .to_numpy()
        .ravel()
    )
    llos[s] = (
        pl.scan_parquet(data_dirs[s].joinpath("outcomes.parquet"))
        .select("long_length_of_stay")
        .collect()
        .to_numpy()
        .ravel()
    )

for s in splits:
    assert feats[s].shape[0] == mort[s].shape[0]


clf = skl_ens.IsolationForest(
    random_state=42
)  # "Returns -1 for outliers and 1 for inliers"
out = dict()
out["train"] = clf.fit_predict(feats["train"])
logger.info(
    "train: {n} ({pct:.2f}%) outliers".format(
        n=(out["train"] == -1).sum(), pct=100 * (out["train"] == -1).mean()
    )
)
for s in ("val", "test"):
    out[s] = clf.predict(feats[s])
    logger.info(
        "{s}: {n} ({pct:.2f}%) outliers".format(
            s=s, n=(out[s] == -1).sum(), pct=100 * (out[s] == -1).mean()
        )
    )

for outcome in ("mortality", "long_los"):
    logger.info(outcome)
    for s in splits:
        logger.info(s)
        xtab = (
            pl.DataFrame(
                {
                    "mortality": mort[s],
                    "long_los": llos[s],
                    "outlier_24h": out[s] == -1,
                }
            )
            .group_by([outcome, "outlier_24h"])
            .agg(pl.len().alias("count"))
            .pivot(values="count", index="outlier_24h", on=outcome, sort_columns=True)
        )
        logger.info(xtab)

        xtab_in = xtab.row(by_predicate=~pl.col("outlier_24h"), named=True)
        in_outcome_rate = xtab_in["true"] / (xtab_in["true"] + xtab_in["false"])
        logger.info("inlier mortality rate: {:.2f}%".format(100 * in_outcome_rate))

        xtab_out = xtab.row(by_predicate=pl.col("outlier_24h"), named=True)
        out_outcome_rate = xtab_out["true"] / (xtab_out["true"] + xtab_out["false"])
        logger.info("outlier mortality rate: {:.2f}%".format(100 * out_outcome_rate))

        logger.info("risk factor: {:.2f}".format(out_outcome_rate / in_outcome_rate))


# skl_mfld.Isomap(n_jobs=-1) supports transformation of out-of-sample data but is slow
pca = skl_decomp.PCA(n_components=2)
embd = dict()
embd["train"] = pca.fit_transform(feats["train"])
logger.info(f"{pca.explained_variance_ratio_=}")

for s in ("val", "test"):
    embd[s] = pca.transform(feats[s])

df = pl.concat(
    pl.DataFrame(
        {
            "pca_0": embd[s][:, 0],
            "pca_1": embd[s][:, 1],
            "split": s,
            "mortality": mort[s],
            "long_los": llos[s],
            "outlier_24h": out[s] == -1,
        }
    )
    for s in splits
)


for outcome in ("mortality", "long_los"):
    fig = px.scatter(
        df.sort(outcome),
        x="pca_0",
        y="pca_1",
        color=outcome,
        symbol="split",
        title=f"Representation embedding colored by {outcome}",
        opacity=0.1,
    )
    fig.update_layout(
        xaxis_title="1st PCA (expl. var. ratio = {:.2f})".format(
            pca.explained_variance_ratio_[0]
        ),
        yaxis_title="2nd PCA (expl. var. ratio = {:.2f})".format(
            pca.explained_variance_ratio_[1]
        ),
    )
    fig.write_html(
        out_dir.joinpath("reps-pca-{m}-{o}.html".format(m=model_loc.stem, o=outcome))
    )
