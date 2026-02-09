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
import sklvq

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
parser.add_argument("--k", type=int, default=25)
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

Xtrain = features["orig"]["train"]
ytrain = labels["orig"]["train"]
Xval = features["orig"]["val"]
yval = labels["orig"]["val"]

scaler = skl.pipeline.make_pipeline(
    skl.preprocessing.StandardScaler(),
    skl.decomposition.PCA(n_components=args.k, random_state=42),
)
Xtrain_p = scaler.fit_transform(Xtrain)

# classes, ytrain_i = np.unique(ytrain, return_inverse=True)
# protoype_n_per_class = np.clip(
#     [20 * (ytrain == c).sum() // len(ytrain) for c in classes], a_min=1, a_max=10
# )

model = sklvq.GMLVQ(random_state=42, solver_type="lbfgs", prototype_n_per_class=3)
model.fit(Xtrain_p, ytrain)

outcome = "long_length_of_stay"
mdl = skl.mixture.GaussianMixture(n_components=1)
flag = np.char.find(ytrain.astype(str), outcome) > -1
bics = {
    n: skl.mixture.GaussianMixture(n_components=n)
    .fit(Xtrain[flag])
    .bic(Xval[np.char.find(yval.astype(str), outcome) > -1])
    for n in range(1, 11)
}

n_optimal = min(bics.keys(), key=bics.get)

for v in versions:
    ytest = labels[v]["test"]
    Xtest_p = scaler.transform(features[v]["test"])
    ytest_pred = model.predict(Xtest_p)
    logger.info(skl.metrics.classification_report(ytest, ytest_pred))
    ytest_probs = model.predict_proba(Xtest_p)
    for outcome in args.outcomes:
        logger.info(v.upper() + " " + outcome.upper().ljust(79, "-"))
        ix = np.char.find(model.classes_.astype(str), outcome) > -1
        class_preds = ytest_probs[:, ix].sum(axis=1)
        class_trues = np.char.find(ytest.astype(str), outcome) > -1
        auc = skl.metrics.roc_auc_score(class_trues, class_preds)
        logger.info(f"{auc=:.3f}")

# omega = model.get_omega()
# protos = model.get_prototypes()
# p_labels = model.prototypes_labels_
# model.classes_
# np.testing.assert_allclose(omega.T @ omega, model.lambda_)

if args.save_params:
    with open(
        data_dirs["orig"]["train"] / ("gmlvq-" + model_loc.stem + ".pkl"), "wb"
    ) as fp:
        pickle.dump({"scaler": scaler, "model": model}, fp)
        fix_perms(fp)

logger.info("---fin")
