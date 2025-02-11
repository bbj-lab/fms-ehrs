#!/usr/bin/env python3

"""
make some simple predictions outcomes ~ features
break down performance by ICU admission type
"""

import os
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import sklearn.metrics as skl_mets

pd.options.display.float_format = "{:,.3f}".format

data_version = "day_stays_qc_first_24h"
model_version = "small-lr-search"  # "small"
# set the following flag to "False" for better performance
fast = False

if os.uname().nodename.startswith("cri"):
    hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-data")
else:
    # change following line to develop locally
    hm = pathlib.Path(__file__).parent.joinpath("results").absolute()

splits = ("train", "val", "test")
data_dirs = dict()
for s in splits:
    data_dirs[s] = hm.joinpath(f"{data_version}-tokenized", s)

""" classifier for admission mortality
"""
data = dict()
for s in ("train", "val"):
    feats = np.load(data_dirs[s].joinpath("features-{m}.npy".format(m=model_version)))
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
    100 if fast else 1000,
    valid_sets=[data["val"]],
)
mort_pred = bst.predict(np.load(data_dirs["test"].joinpath("features.npy")))
mort_true = (
    pl.scan_parquet(data_dirs["test"].joinpath("outcomes.parquet"))
    .select("same_admission_death")
    .cast(pl.Int64)
    .collect()
    .to_numpy()
)

print(
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
    print(
        "{}: {:.3f}".format(
            met,
            getattr(skl_mets, f"{met}_score")(
                y_true=mort_true, y_pred=np.round(mort_pred)
            ),
        )
    )

""" regression for length of stay (in hours)
"""

for s in ("train", "val"):
    feats = np.load(data_dirs[s].joinpath("features-{m}.npy".format(m=model_version)))
    mort = (
        pl.scan_parquet(data_dirs[s].joinpath("outcomes.parquet"))
        .select("length_of_stay")
        .collect()
        .to_numpy()
        .ravel()
    )
    data[s] = lgb.Dataset(feats, label=mort)


bst = lgb.train(
    {"objective": "regression", "force_col_wise": True}
    | ({} if fast else {"learning_rate": 0.05}),
    data["train"],
    100 if fast else 1000,
    valid_sets=[data["val"]],
)
los_pred = bst.predict(np.load(data_dirs["test"].joinpath("features.npy")))
los_true = (
    pl.scan_parquet(data_dirs["test"].joinpath("outcomes.parquet"))
    .select("length_of_stay")
    .cast(pl.Int64)
    .collect()
    .to_numpy()
)

for met in (
    "root_mean_squared_error",
    "mean_absolute_error",
    "explained_variance_score",
):
    print(
        "{}: {:.3f}".format(
            met,
            getattr(skl_mets, met)(y_true=los_true, y_pred=los_pred),
        )
    )


if os.uname().nodename.startswith("cri"):

    test_ids = pl.scan_parquet(
        data_dirs["test"].joinpath("tokens_timelines.parquet")
    ).select("hospitalization_id")
    icu_ids = pl.scan_csv(hm.parent.joinpath("mimic_icu_types.csv.gz")).cast(
        {"hadm_id": pl.String}
    )
    test_icu = test_ids.join(
        icu_ids,
        how="left",
        left_on="hospitalization_id",
        right_on="hadm_id",
        maintain_order="left",  # neat polars feature
    )

    icu_mask = (
        test_icu.select(~pl.col("careunit").is_null()).collect().to_numpy().ravel()
    )

    print(
        "ICU roc_auc: {:.3f}".format(
            skl_mets.roc_auc_score(
                y_true=mort_true[icu_mask], y_score=mort_pred[icu_mask]
            )
        )
    )

    print(
        "No ICU roc_auc: {:.3f}".format(
            skl_mets.roc_auc_score(
                y_true=mort_true[~icu_mask], y_score=mort_pred[~icu_mask]
            )
        )
    )

    icu_types = (
        test_icu.select("careunit").drop_nulls().unique().collect().to_numpy().ravel()
    )
    results = pd.DataFrame(
        columns=[
            "count",
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
        ],
        index=icu_types,
    ).rename_axis(index="icu_types")

    for icu_t in icu_types:
        icu_t_mask = (
            test_icu.select(pl.col("careunit") == icu_t)
            .fill_null(False)
            .collect()
            .to_numpy()
            .ravel()
        )
        results.loc[icu_t, "count"] = icu_t_mask.sum()
        results.loc[icu_t, "roc_auc"] = skl_mets.roc_auc_score(
            y_true=mort_true[icu_t_mask], y_score=mort_pred[icu_t_mask]
        )
        for met in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
        ):
            results.loc[icu_t, met] = getattr(skl_mets, f"{met}_score")(
                y_true=mort_true[icu_t_mask], y_pred=np.round(mort_pred[icu_t_mask])
            )

    print(results.astype({"count": "int"}))
