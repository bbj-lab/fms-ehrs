#!/usr/bin/env python3

"""
learn prototypes for multi-class classification outcomes
"""

import argparse
import collections
import pathlib
import pickle

import numpy as np
import polars as pl
import sklearn as skl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import fix_perms
from fms_ehrs.framework.util import set_pd_options

set_pd_options()

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../data-ucmc")
parser.add_argument("--data_version", type=str, default="Y21_first_24h")
parser.add_argument(
    "--model_loc", type=pathlib.Path, default="../../mdls-archive/gemma-5635921-Y21"
)
parser.add_argument(
    "--outcomes",
    nargs="+",
    default=[
        "same_admission_death",
        "long_length_of_stay",
        # "ama_discharge",
        # "hospice_discharge",
    ],
)
parser.add_argument("--save_params", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc),
)

splits = ("train", "val", "test")
versions = ("orig", "new")

data_dirs = collections.defaultdict(dict)
features = collections.defaultdict(dict)
labels = collections.defaultdict(dict)

for v in versions:
    for s in splits:
        data_dirs[v][s] = (
            (data_dir_orig if v == "orig" else data_dir_new)
            / f"{args.data_version}-tokenized"
            / s
        )
        features[v][s] = np.load(
            data_dirs[v][s] / "features-{m}.npy".format(m=model_loc.stem)
        )
        for outcome in args.outcomes:
            labels[v][s] = (
                pl.scan_parquet(data_dirs[v][s] / "tokens_timelines_outcomes.parquet")
                .select(
                    pl.concat_str(
                        [
                            pl.when(pl.col(outcome) == 1)
                            .then(pl.lit(outcome))
                            .otherwise(None)
                            for outcome in args.outcomes
                        ],
                        separator=",",
                        ignore_nulls=True,
                    ).replace("", "none")
                )
                .collect()
                .to_numpy()
                .ravel()
            )


scaler = skl.discriminant_analysis.LinearDiscriminantAnalysis()
Xtrain = scaler.fit_transform(features["orig"]["train"], labels["orig"]["train"])
ytrain = labels["orig"]["train"]
Xval = scaler.transform(features["orig"]["val"])
yval = labels["orig"]["val"]


models = dict()
for outcome in args.outcomes + ["none"]:
    Xto = Xtrain[np.char.find(ytrain.astype(str), outcome) > -1]
    Xvo = Xval[np.char.find(yval.astype(str), outcome) > -1]
    bics = {
        n: skl.mixture.GaussianMixture(n_components=n, init_params="k-means++")
        .fit(Xto)
        .bic(Xvo)
        for n in range(1, 20)
    }
    n_optimal = min(bics.keys(), key=bics.get)
    models[outcome] = skl.mixture.GaussianMixture(
        n_components=n_optimal, init_params="k-means++"
    ).fit(Xto)


for v in versions:
    for outcome in args.outcomes:
        n0 = np.sum(np.char.find(ytrain.astype(str), outcome) > -1)
        n1 = np.sum(ytrain == "none")
        prevalence = n0 / (n0 + n1)
        Xtest = scaler.transform(features[v]["test"])
        combined_model = skl.mixture.GaussianMixture(
            n_components=models[outcome].n_components + models["none"].n_components
        )
        combined_model.weights_ = np.concatenate(
            [
                models[outcome].weights_ * prevalence,
                models["none"].weights_ * (1 - prevalence),
            ]
        )
        for att in ("means_", "covariances_", "precisions_", "precisions_cholesky_"):
            setattr(
                combined_model,
                att,
                np.concatenate(
                    [getattr(models[outcome], att), getattr(models["none"], att)]
                ),
            )
        combined_model.n_features_in_ = models[outcome].n_features_in_
        component_labels = np.array(
            [outcome] * models[outcome].n_components
            + ["none"] * models["none"].n_components
        )
        component_preds = combined_model.predict_proba(Xtest)
        ytest = labels[v]["test"]
        yeo = np.char.find(ytest.astype(str), outcome) > -1
        ypred_proba = component_preds[:, component_labels == outcome].sum(axis=1)
        auc = skl.metrics.roc_auc_score(yeo, ypred_proba)
        logger.info(f"{v=}, {outcome=}, {auc=:.3f}")


if args.save_params:
    with open(
        data_dirs["orig"]["train"] / ("lda-gmm-protos-" + model_loc.stem + ".pkl"), "wb"
    ) as fp:
        pickle.dump({"scaler": scaler, "models": models}, fp)
        fix_perms(fp)

logger.info("---fin")

"""
[2026-02-10T16:20:56CST] v='orig', outcome='same_admission_death', auc=0.868
[2026-02-10T16:20:57CST] v='orig', outcome='long_length_of_stay', auc=0.751
[2026-02-10T16:20:58CST] v='new', outcome='same_admission_death', auc=0.842
[2026-02-10T16:21:00CST] v='new', outcome='long_length_of_stay', auc=0.651
"""
