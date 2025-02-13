#!/usr/bin/env python3

"""
compare predictions form XGB classification to the FHE version
"""

import os
import pathlib

import numpy as np
import polars as pl

from xgboost.sklearn import XGBClassifier as SklearnXGBClassifier
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier
from concrete.compiler import check_gpu_available
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.model_selection import GridSearchCV


if os.uname().nodename.startswith("cri"):
    data_hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-data")
else:
    # change following line to develop locally
    data_hm = pathlib.Path(__file__).parent.joinpath("results").absolute()


data_version = "day_stays_qc_first_24h"
model_version = "smaller-lr-search"
n_estimators = 10
max_depth = 6
n_bits = 6

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
        .ravel()
    )

"""
basic comparison
"""

xgb_sklearn = SklearnXGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    n_jobs=-1,
)
xgb_sklearn.fit(feats["train"], mort["train"], eval_set=[(feats["val"], mort["val"])])
mort_xgb_pred = xgb_sklearn.predict_proba(feats["test"])[:, 1]
print(
    "Vanilla XGB AUC: {:.3f}".format(
        skl_auc(y_true=mort["test"], y_score=mort_xgb_pred)
    )
)

cml_sklearn = ConcreteXGBClassifier(
    n_bits=n_bits,
    n_estimators=n_estimators,
    max_depth=max_depth,
    n_jobs=-1,
)
cml_sklearn.fit(feats["train"], mort["train"], eval_set=[(feats["val"], mort["val"])])
mort_cml_pred = cml_sklearn.predict_proba(feats["test"])[:, 1]
print(
    "Concrete ML AUC: {:.3f}".format(
        skl_auc(y_true=mort["test"], y_score=mort_cml_pred)
    )
)

"""
hyperparameter tuning
"""

param_grid = {
    "max_depth": [4, 6],
    "n_estimators": [10, 20],
}

sklearn_grid_search = GridSearchCV(
    SklearnXGBClassifier(), param_grid, cv=3, verbose=True, n_jobs=-1
)
sklearn_grid_search.fit(
    X=np.concatenate([feats["train"], feats["val"]]),
    y=np.concatenate([mort["train"], mort["val"]]),
)
mort_xgb_hp_pred = sklearn_grid_search.best_estimator_.predict_proba(feats["test"])[
    :, 1
]
print(
    "Vanilla XGB AUC (HP-tuned): {:.3f}".format(
        skl_auc(y_true=mort["test"], y_score=mort_xgb_hp_pred)
    )
)

cml_grid_search = GridSearchCV(
    ConcreteXGBClassifier(),
    param_grid | {"n_bits": [4, 6, 8]},
    cv=3,
    verbose=True,
    n_jobs=-1,
)
cml_grid_search.fit(
    X=np.concatenate([feats["train"], feats["val"]]),
    y=np.concatenate([mort["train"], mort["val"]]),
)
mort_cml_hp_pred = cml_grid_search.best_estimator_.predict_proba(feats["test"])[:, 1]
print(
    "Concrete ML AUC (HP-tuned): {:.3f}".format(
        skl_auc(y_true=mort["test"], y_score=mort_cml_hp_pred)
    )
)

"""
inference with FHE
"""

device = "cuda" if check_gpu_available() else "cpu"
concrete_model = ConcreteXGBClassifier(
    n_estimators=n_estimators, max_depth=max_depth
)  # **cml_grid_search.best_params_
concrete_model.fit(
    feats["train"],
    mort["train"],
    eval_set=[(feats["val"], mort["val"])],
)
circuit = concrete_model.compile(feats["train"], device=device)
circuit.client.keygen(force=False)
y_preds_fhe = concrete_model.predict_proba(feats["test"][:10], fhe="execute")[
    :, 1
]  # first 10 only; this is slow
