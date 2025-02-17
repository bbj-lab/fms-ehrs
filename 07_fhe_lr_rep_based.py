#!/usr/bin/env python3

"""
compare predictions form XGB classification to the FHE version
"""

import os
import pathlib

import numpy as np
import polars as pl
from concrete.ml.sklearn import SGDClassifier

if os.uname().nodename.startswith("cri"):
    data_hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-data")
else:
    # change following line to develop locally
    data_hm = pathlib.Path(__file__).parent.joinpath("results").absolute()


data_version = "day_stays_qc_first_24h"
model_version = "smaller-lr-search"

splits = ("train", "val", "test")
feats = dict()
mort = dict()

for s in splits:
    feats[s] = np.load(
        data_hm.joinpath(
            f"{data_version}-tokenized", s, "features-{m}.npy".format(m=model_version)
        )
    )
    mort[s] = (
        pl.scan_parquet(
            data_hm.joinpath(f"{data_version}-tokenized", s, "outcomes.parquet")
        )
        .select("same_admission_death")
        .collect()
        .to_numpy()
        .astype(int)
        .ravel()
    )

"""
training on encrypted data
"""

parameters_range = (-1.0, 1.0)
mn, mx = feats["train"].min(axis=0), feats["train"].max(axis=0)
feats["train"] = (2 * feats["train"] - mx - mn) / (mx - mn)
for s in ("val", "test"):
    feats[s] = np.clip(
        (2 * feats[s] - mx - mn) / (mx - mn),
        a_min=min(parameters_range),
        a_max=max(parameters_range),
    )

# weird that we can't seem to break encryption and model training into separate
# steps here -- we would want / need to do this for most practical use cases

model = SGDClassifier(
    random_state=42,
    max_iter=10,
    fit_encrypted=True,
    parameters_range=parameters_range,
    verbose=1,
)

n_train = 100
model.fit(feats["train"][:n_train], mort["train"][:n_train], fhe="execute")
model.compile(feats["train"][:n_train])

model.predict_proba(feats["val"][:10], fhe="execute")

""" typical output on local machine:
Compiling training circuit on device 'cpu'...
Compilation took 2.2327 seconds.
Key Generation...
Key generation took 19.0704 seconds.
Training on encrypted data...
Iteration 0 took 303.54 seconds.
Iteration 1 took 293.87 seconds.
Iteration 2 took 321.67 seconds.
Iteration 3 took 292.35 seconds.
"""
