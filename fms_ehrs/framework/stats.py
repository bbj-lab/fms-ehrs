#!/usr/bin/env python3

"""
statistical & bootstrapping-related functions
"""

import typing
import warnings

import joblib as jl
import numpy as np
from sklearn import metrics as skl_mets

Generator: typing.TypeAlias = np.random._generator.Generator


def boostrap_roc_auc_ci(
    y_true: np.array,
    y_score: np.array,
    *,
    n_samples: int = 10_000,
    alpha: float = 0.05,
    rng: Generator = np.random.default_rng(seed=42),
    n_jobs: int = -1,
    **auc_kwargs,
):
    """
    Calculates a bootstrapped percentile interval for AUC as described in §13.3
    of Efron & Tibshirani's "An Introduction to the Bootstrap" (Chapman & Hall,
    Boca Raton, 1993), ignoring variance due to model-fitting (i.e. a 'liberal'
    bootstrap for variability in the test-set alone)
    """

    def score_i(rng_i: Generator) -> float:
        warnings.filterwarnings("ignore")
        return skl_mets.roc_auc_score(
            y_true[samp_i := rng_i.choice(len(y_true), size=len(y_true), replace=True)],
            y_score[samp_i],
            **auc_kwargs,
        )

    with jl.Parallel(n_jobs=n_jobs) as par:
        scores = par(jl.delayed(score_i)(rng_i) for rng_i in rng.spawn(n_samples))

    return np.quantile(scores, q=[alpha / 2, 1 - (alpha / 2)])


def bootstrap_test_roc_auc_pval(
    y_true: np.array,
    y_score0: np.array,
    y_score1: np.array,
    *,
    n_samples: int = 10_000,
    rng: Generator = np.random.default_rng(seed=42),
    n_jobs: int = -1,
    alternative: typing.Literal["one-sided", "two-sided"] = "one-sided",
    **auc_kwargs,
):
    """
    Performs a bootstrapped test for the null hypothesis that `y_score0` &
    `y_score1` are equally good predictions of y_true (in terms of AUC), as
    outlined in Algorithm 16.1 of Efron & Tibshirani's "An Introduction to the
    Bootstrap" (Chapman & Hall, Boca Raton, 1993), ignoring variance due to
    model-fitting (i.e. a 'liberal' bootstrap for variability in the test-set
    alone)
    """
    auc0 = skl_mets.roc_auc_score(y_true, y_score0, **auc_kwargs)
    auc1 = skl_mets.roc_auc_score(y_true, y_score1, **auc_kwargs)
    diff_obs = auc1 - auc0

    y_trues = np.concatenate([y_true, y_true])
    y_scores = np.concatenate([y_score0, y_score1])

    def diff_i(rng_i: Generator) -> float:
        warnings.filterwarnings("ignore")
        auc0_i = skl_mets.roc_auc_score(
            y_trues[
                samp0_i := rng_i.choice(len(y_trues), size=len(y_true), replace=True)
            ],
            y_scores[samp0_i],
            **auc_kwargs,
        )
        auc1_i = skl_mets.roc_auc_score(
            y_trues[
                samp1_i := rng_i.choice(len(y_trues), size=len(y_true), replace=True)
            ],
            y_scores[samp1_i],
            **auc_kwargs,
        )
        return auc1_i - auc0_i

    with jl.Parallel(n_jobs=n_jobs) as par:
        diffs = par(jl.delayed(diff_i)(rng_i) for rng_i in rng.spawn(n_samples))

    return (
        np.mean(diffs > diff_obs)
        if alternative == "one-sided"
        else np.mean(np.abs(diffs) > abs(diff_obs))
    ).item()


def generate_classifier_preds(
    n: int = 1000,
    num_preds: int = 1,
    frac_1: float = 0.8,
    rng: Generator = np.random.default_rng(seed=42),
):
    assert 0 <= frac_1 <= 1
    y_seed = rng.uniform(size=n)
    y_true = (y_seed > 1 - frac_1).astype(int)

    y_preds = [
        np.clip(
            y_seed + rng.normal(scale=(2 * i + 5) / 27, size=1000), a_min=0, a_max=1
        )
        for i in range(num_preds)
    ]

    return y_true, y_preds


if __name__ == "__main__":

    np_rng = np.random.default_rng(42)

    y_true, y_preds = generate_classifier_preds(num_preds=2)

    for i in range(len(y_preds)):
        print(
            "AUC for preds{} = {:.3f}".format(
                i,
                skl_mets.roc_auc_score(y_true=y_true, y_score=y_preds[i]),
            )
        )
        print(
            "CI  for preds{} = {}".format(
                i,
                boostrap_roc_auc_ci(y_true=y_true, y_score=y_preds[i]).round(3),
            )
        )

    print(
        "test 1-sided: p = {:.3f}".format(
            bootstrap_test_roc_auc_pval(
                y_true=y_true, y_score0=y_preds[1], y_score1=y_preds[0]
            ),
        )
    )

    print(
        "test 2-sided: p = {:.3f}".format(
            bootstrap_test_roc_auc_pval(
                y_true=y_true,
                y_score0=y_preds[1],
                y_score1=y_preds[0],
                alternative="two-sided",
            ),
        )
    )
