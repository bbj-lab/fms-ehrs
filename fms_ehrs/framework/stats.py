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


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_samples: int = 10_000,
    alpha: float = 0.05,
    rng: Generator = np.random.default_rng(seed=42),
    objs: typing.Tuple[typing.Literal["roc_auc", "pr_auc", "brier"], ...] = (
        "roc_auc",
        "pr_auc",
        "brier",
    ),
    n_jobs: int = -1,
) -> dict:
    """
    Calculates a bootstrapped percentile interval for objectives `objs` as
    described in §13.3 of Efron & Tibshirani's "An Introduction to the Bootstrap"
    (Chapman & Hall, Boca Raton, 1993), ignoring variance due to model-fitting
    (i.e. a 'liberal' bootstrap for variability in the test-set alone)
    """

    def get_scores_i(rng_i: Generator) -> dict[str, float]:
        warnings.filterwarnings("ignore")
        yti = y_true[
            samp_i := rng_i.choice(len(y_true), size=len(y_true), replace=True)
        ]
        ysi = y_score[samp_i]
        ret = dict()
        if "roc_auc" in objs:
            ret["roc_auc"] = skl_mets.roc_auc_score(yti, ysi)
        if "pr_auc" in objs:
            precs, recs, _ = skl_mets.precision_recall_curve(
                yti, np.round(ysi, decimals=4), drop_intermediate=True
            )
            ret["pr_auc"] = skl_mets.auc(recs, precs)
        if "brier" in objs:
            ret["brier"] = skl_mets.brier_score_loss(yti, ysi)
        return ret

    with jl.Parallel(n_jobs=n_jobs) as par:
        scores = par(jl.delayed(get_scores_i)(rng_i) for rng_i in rng.spawn(n_samples))

    return {
        ob: np.quantile([s[ob] for s in scores], q=[alpha / 2, 1 - (alpha / 2)])
        for ob in objs
    }


def bootstrap_pval(
    y_true: np.ndarray,
    y_score0: np.ndarray,
    y_score1: np.ndarray,
    *,
    n_samples: int = 10_000,
    rng: Generator = np.random.default_rng(seed=42),
    alternative: typing.Literal["one-sided", "two-sided"] = "one-sided",
    objs: typing.Tuple[typing.Literal["roc_auc", "pr_auc", "brier"], ...] = (
        "roc_auc",
        "pr_auc",
        "brier",
    ),
    n_jobs: int = -1,
) -> dict:
    """
    Performs a bootstrapped test for the null hypothesis that `y_score0` &
    `y_score1` are equally good predictions of y_true (in terms of `objs`), as
    outlined in Algorithm 16.1 of Efron & Tibshirani's "An Introduction to the
    Bootstrap" (Chapman & Hall, Boca Raton, 1993), ignoring variance due to
    model-fitting (i.e. a 'liberal' bootstrap for variability in the test-set
    alone); one-sided alternative corresponds to`y_score1` being better than
    `y_score0`
    """

    def get_diffs(yt0, ys0, yt1, ys1) -> dict[str, float]:
        diffs = dict()
        if "roc_auc" in objs:
            diffs["roc_auc"] = skl_mets.roc_auc_score(
                yt1, ys1
            ) - skl_mets.roc_auc_score(yt0, ys0)
        if "pr_auc" in objs:
            precs1, recs1, _ = skl_mets.precision_recall_curve(
                yt1, np.round(ys1, decimals=4), drop_intermediate=True
            )
            precs0, recs0, _ = skl_mets.precision_recall_curve(
                yt0, np.round(ys0, decimals=4), drop_intermediate=True
            )
            diffs["pr_auc"] = skl_mets.auc(recs1, precs1) - skl_mets.auc(recs0, precs0)
        if "brier" in objs:  # higher brier is worse
            diffs["brier"] = -1 * (
                skl_mets.brier_score_loss(yt1, ys1)
                - skl_mets.brier_score_loss(yt0, ys0)
            )
        return diffs

    diff_obs = get_diffs(y_true, y_score0, y_true, y_score1)

    y_trues = np.concatenate([y_true, y_true])
    y_scores = np.concatenate([y_score0, y_score1])

    def get_diffs_i(rng_i: Generator) -> dict[str, float]:
        return get_diffs(
            y_trues[
                samp0_i := rng_i.choice(len(y_trues), size=len(y_true), replace=True)
            ],
            y_scores[samp0_i],
            y_trues[
                samp1_i := rng_i.choice(len(y_trues), size=len(y_true), replace=True)
            ],
            y_scores[samp1_i],
        )

    with jl.Parallel(n_jobs=n_jobs) as par:
        diffs = par(jl.delayed(get_diffs_i)(rng_i) for rng_i in rng.spawn(n_samples))

    if alternative == "one-sided":
        return {ob: np.mean([d[ob] > diff_obs[ob] for d in diffs]) for ob in objs}
    else:  # two-sided
        return {
            ob: np.mean([np.abs(d[ob]) > np.abs(diff_obs[ob]) for d in diffs])
            for ob in objs
        }


def generate_classifier_preds(
    n: int = 1000,
    num_preds: int = 1,
    frac_1: float = 0.8,
    rng: Generator = np.random.default_rng(seed=42),
) -> tuple[np.ndarray, np.ndarray]:
    assert 0 <= frac_1 <= 1
    y_seed = rng.uniform(size=n)
    y_true = (y_seed > 1 - frac_1).astype(int)

    y_preds = np.array(
        [
            np.clip(
                y_seed + rng.normal(scale=(2 * i + 5) / 27, size=1000), a_min=0, a_max=1
            )
            for i in range(num_preds)
        ]
    )

    if num_preds > 2:
        y_preds[-1] = 0.975 * y_preds[-1]

    return y_true, y_preds


if __name__ == "__main__":
    np_rng = np.random.default_rng(42)

    y_true, y_preds = generate_classifier_preds(num_preds=3)

    for i in range(len(y_preds)):
        print(
            "AUC for preds{} = {:.3f}".format(
                i, skl_mets.roc_auc_score(y_true=y_true, y_score=y_preds[i])
            )
        )
        print(
            "CI  for preds{} = {}".format(
                i,
                bootstrap_ci(
                    y_true=y_true,
                    y_score=y_preds[i],
                    objs=("roc_auc",),
                    n_samples=1_000,
                )["roc_auc"].round(3),
            )
        )
        print("all metrics: {}".format(bootstrap_ci(y_true=y_true, y_score=y_preds[i])))

    print(
        "test 0 vs 1, 1-sided: {}".format(
            bootstrap_pval(
                y_true=y_true, y_score0=y_preds[1], y_score1=y_preds[0], n_samples=1_000
            )
        )
    )

    print(
        "test 0 vs 1, 2-sided: {}".format(
            bootstrap_pval(
                y_true=y_true,
                y_score0=y_preds[1],
                y_score1=y_preds[0],
                n_samples=1_000,
                alternative="two-sided",
            )
        )
    )

    print(
        "test 0 vs 2, 1-sided: {}".format(
            bootstrap_pval(
                y_true=y_true, y_score0=y_preds[2], y_score1=y_preds[0], n_samples=1_000
            )
        )
    )

    print(
        "test 0 vs 2, 2-sided: {}".format(
            bootstrap_pval(
                y_true=y_true,
                y_score0=y_preds[2],
                y_score1=y_preds[0],
                n_samples=1_000,
                alternative="two-sided",
            )
        )
    )
