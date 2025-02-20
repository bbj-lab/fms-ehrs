#!/usr/bin/env python3

"""
collect n_samp repetitions of vllm predictions and process them
"""

import pathlib
import pickle

import numpy as np
import polars as pl
import sklearn.metrics as skl_mets

from vocabulary import Vocabulary

data_version = "day_stays_qc_first_24h"
model_version = "small-packed"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()
k = 25_000

# load and prep data
splits = ("train", "val", "test")
data_dirs = {
    s: hm.joinpath("clif-data", f"{data_version}-tokenized", s) for s in splits
}

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

determined = list()
expired = list()
n_gend = list()
fin_tkn = list()

for fp in data_dirs["test"].glob(f"responses_k{k}_rep_*_of_*-{model_version}.pkl"):
    det = list()
    exp = list()
    gen = list()
    fin = list()
    with open(fp, "rb") as f:
        response_list = pickle.load(f)
        for x in response_list:
            if x[-1] not in {vocab("expired"), vocab("TL_END")}:
                print(x[-1])

            det.append(len(x) < k)
            exp.append(vocab("expired") in x)
            gen.append(len(x))
            fin.append(x[-1])
    determined.append(det)
    expired.append(exp)
    n_gend.append(gen)
    fin_tkn.append(fin)

determined, expired, n_gend, fin_tkn = map(
    np.array, (determined, expired, n_gend, fin_tkn)
)

print("Determined: {}%".format(100 * determined.mean().round(5)))
print(
    "Tokens generated: {:.2f} avg. (std. of {:.2f})".format(n_gend.mean(), n_gend.std())
)

mort_pred = expired.mean(axis=0)

mort_true = (
    pl.scan_parquet(data_dirs["test"].joinpath("outcomes.parquet"))
    .select("same_admission_death")
    .cast(pl.Int64)
    .collect()
    .to_numpy()
)

print(
    "{d} deaths in population of {pop} ({pct:.2f}%)".format(
        d=mort_true.sum(),
        pop=mort_true.size,
        pct=mort_true.mean() * 100,
    ),
)

print("{pred} predicted deaths".format(pred=mort_pred.round().astype(int).sum()))

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
                y_true=mort_true, y_pred=(mort_pred >= 0.5)
            ),
        )
    )

uniq = lambda arr: dict(zip(*np.unique(arr, return_counts=True)))
for k, v in uniq(fin_tkn.ravel()).items():
    print(vocab.reverse[k], ": ", np.round(v / fin_tkn.size, 3), sep="")
