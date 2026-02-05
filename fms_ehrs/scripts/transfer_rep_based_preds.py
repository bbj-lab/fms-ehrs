#!/usr/bin/env python3

"""
make some simple predictions outcomes ~ features
provide some performance breakdowns
"""

import argparse
import collections
import pathlib
import pickle
import typing

import lightgbm as lgb
import numpy as np
import polars as pl
import sklearn as skl

from fms_ehrs.framework.logger import get_logger, log_classification_metrics
from fms_ehrs.framework.storage import fix_perms
from fms_ehrs.framework.util import set_pd_options

set_pd_options()

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


def _parse_float_list(s: str) -> list[float]:
    # Accept either "[0.1,1,10]" or "0.1,1,10"
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _ece(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 15) -> float:
    """Expected calibration error (ECE) for probabilistic binary predictions."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (y_prob >= lo) & (y_prob <= hi)
        else:
            m = (y_prob >= lo) & (y_prob < hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y_true[m]))
        conf = float(np.mean(y_prob[m]))
        ece += float(np.mean(m)) * abs(acc - conf)
    return float(ece)


def _choose_threshold(
    y_true_val: np.ndarray,
    y_score_val: np.ndarray,
    *,
    strategy: str,
) -> float:
    """Choose a decision threshold on the validation set."""
    y_true_val = np.asarray(y_true_val).astype(int)
    y_score_val = np.asarray(y_score_val).astype(float)
    if strategy == "fixed_0.5":
        return 0.5
    if strategy == "youden_j":
        fpr, tpr, thr = skl.metrics.roc_curve(y_true_val, y_score_val)
        j = tpr - fpr
        return float(thr[int(np.argmax(j))])
    if strategy == "f1":
        prec, rec, thr = skl.metrics.precision_recall_curve(y_true_val, y_score_val)
        # precision_recall_curve returns thresholds of length n-1; align carefully.
        f1 = (2 * prec * rec) / np.clip((prec + rec), 1e-12, None)
        if thr.size == 0:
            return 0.5
        best = int(np.argmax(f1[:-1]))
        return float(thr[best])
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric_fn: typing.Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI for metrics that take (y_true, y_score)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = y_true.shape[0]
    vals: list[float] = []
    tries = 0
    # Some metrics (e.g., AUROC) are undefined if a resample has only one class.
    while len(vals) < n_boot and tries < (n_boot * 10):
        tries += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if np.unique(yt).size < 2:
            continue
        try:
            vals.append(float(metric_fn(yt, ys)))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path)
parser.add_argument("--data_dir_new", type=pathlib.Path)
parser.add_argument("--data_version", type=str)
parser.add_argument("--model_loc", type=pathlib.Path)
parser.add_argument(
    "--classifier",
    choices=["light_gbm", "logistic_regression_cv", "logistic_regression"],
    default="logistic_regression",
)
parser.add_argument("--save_preds", action="store_true")
parser.add_argument("--drop_icu_adm", action="store_true")
parser.add_argument(
    "--outcomes",
    nargs="+",
    default=None,
    help=(
        "Optional explicit list of outcome columns to evaluate (space-separated). "
        "If not provided, defaults to the canonical fms-ehrs set. "
        "This is useful for experiment-specific task sets (e.g., Exp3 H_ICU cohort: LOS>=24h + linked ICU stay)."
    ),
)
parser.add_argument(
    "--tune_logreg_C",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Tune logistic regression regularization strength C on val (matched-budget grid).",
)
parser.add_argument(
    "--logreg_C_grid",
    type=str,
    default="[0.01,0.1,1,10,100]",
    help="Grid of C values for val tuning (matched-budget across conditions).",
)
parser.add_argument(
    "--threshold_strategy",
    type=str,
    choices=["fixed_0.5", "youden_j", "f1"],
    default="youden_j",
    help="Choose decision threshold on val (AUROC remains threshold-free).",
)
parser.add_argument(
    "--bootstrap_n",
    type=int,
    default=1000,
    help="Number of bootstrap resamples for confidence intervals (0 disables).",
)
parser.add_argument(
    "--bootstrap_seed",
    type=int,
    default=123,
    help="RNG seed for bootstrap resampling.",
)
parser.add_argument(
    "--calibration_bins",
    type=int,
    default=15,
    help="Number of bins for ECE.",
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc),
)

splits = ("train", "val", "test")
versions = ("orig", "new")
if args.outcomes is None:
    outcomes = ("same_admission_death", "long_length_of_stay", "imv_event") + (
        ("icu_admission",) if not args.drop_icu_adm else ()
    )
else:
    outcomes = tuple(str(x) for x in args.outcomes if str(x).strip())
    if args.drop_icu_adm:
        outcomes = tuple(o for o in outcomes if o != "icu_admission")
    if not outcomes:
        raise ValueError("No outcomes specified (empty --outcomes after filtering).")

data_dirs = collections.defaultdict(dict)
features = collections.defaultdict(dict)
qualifiers = collections.defaultdict(lambda: collections.defaultdict(dict))
labels = collections.defaultdict(lambda: collections.defaultdict(dict))

for v in versions:
    for s in splits:
        data_dirs[v][s] = (data_dir_orig if v == "orig" else data_dir_new).joinpath(
            f"{args.data_version}-tokenized", s
        )
        features[v][s] = np.load(
            data_dirs[v][s].joinpath("features-{m}.npy".format(m=model_loc.stem))
        )
        for outcome in outcomes:
            labels[outcome][v][s] = (
                pl.scan_parquet(
                    data_dirs[v][s].joinpath("tokens_timelines_outcomes.parquet")
                )
                .select(outcome)
                .collect()
                .to_numpy()
                .ravel()
            )
            qualifiers[outcome][v][s] = (
                (
                    ~pl.scan_parquet(
                        data_dirs[v][s].joinpath("tokens_timelines_outcomes.parquet")
                    )
                    .select(outcome + "_24h")
                    .collect()
                    .to_numpy()
                    .ravel()
                )  # *not* people who have had this outcome in the first 24h
                if outcome in ("icu_admission", "imv_event")
                else True * np.ones_like(labels[outcome][v][s])
            )


""" classification outcomes
"""

preds = collections.defaultdict(dict)

for outcome in outcomes:
    logger.info(outcome.replace("_", " ").upper().ljust(79, "-"))

    Xtrain = (features["orig"]["train"])[qualifiers[outcome]["orig"]["train"]]
    ytrain = (labels[outcome]["orig"]["train"])[qualifiers[outcome]["orig"]["train"]]
    Xval = (features["orig"]["val"])[qualifiers[outcome]["orig"]["val"]]
    yval = (labels[outcome]["orig"]["val"])[qualifiers[outcome]["orig"]["val"]]

    match args.classifier:
        case "light_gbm":
            estimator = lgb.LGBMClassifier(metric="auc")
            estimator.fit(X=Xtrain, y=ytrain, eval_set=(Xval, yval))

        case "logistic_regression_cv":
            estimator = skl.pipeline.make_pipeline(
                skl.preprocessing.StandardScaler(),
                skl.linear_model.LogisticRegressionCV(
                    max_iter=10_000,
                    n_jobs=-1,
                    refit=True,
                    random_state=42,
                    solver="newton-cholesky",
                ),
            )
            estimator.fit(X=Xtrain, y=ytrain)

        case "logistic_regression":
            C_grid = _parse_float_list(args.logreg_C_grid)
            if not C_grid:
                C_grid = [1.0]

            def _fit_logreg(C: float):
                est = skl.pipeline.make_pipeline(
                    skl.preprocessing.StandardScaler(),
                    skl.linear_model.LogisticRegression(
                        C=float(C),
                        max_iter=10_000,
                        n_jobs=-1,
                        random_state=42,
                        solver="newton-cholesky",
                    ),
                )
                est.fit(X=Xtrain, y=ytrain)
                return est

            best_C = 1.0
            if args.tune_logreg_C and len(C_grid) > 1:
                best_val = -float("inf")
                for C in C_grid:
                    est = _fit_logreg(C)
                    y_score_val = est.predict_proba(Xval)[:, 1]
                    try:
                        auroc = float(skl.metrics.roc_auc_score(y_true=yval, y_score=y_score_val))
                    except Exception:
                        auroc = -float("inf")
                    if auroc > best_val:
                        best_val = auroc
                        best_C = float(C)
                logger.info(f"Selected logreg C on val (AUROC): C={best_C} (grid={C_grid})")
            else:
                logger.info(f"Using logreg C={best_C} (no tuning; grid={C_grid})")

            estimator = _fit_logreg(best_C)

        case _:
            raise NotImplementedError(
                f"Classifier {args.classifier} is not yet supported."
            )

    # Select decision threshold using validation predictions (for probability-based classifiers).
    chosen_threshold = None
    if args.classifier in ("logistic_regression", "logistic_regression_cv"):
        try:
            y_score_val = estimator.predict_proba(Xval)[:, 1]
            chosen_threshold = _choose_threshold(
                y_true_val=yval,
                y_score_val=y_score_val,
                strategy=args.threshold_strategy,
            )
            logger.info(f"Selected threshold on val: {chosen_threshold:.4f} ({args.threshold_strategy})")
        except Exception as e:
            logger.info(f"Threshold selection failed; falling back to 0.5: {e}")
            chosen_threshold = 0.5
    for v in versions:
        logger.info(v.upper())

        q_test = qualifiers[outcome][v]["test"]
        preds[outcome][v] = estimator.predict_proba((features[v]["test"])[q_test])[:, 1]
        y_true = (labels[outcome][v]["test"])[q_test]
        y_score = preds[outcome][v]

        logger.info("overall performance".upper().ljust(49, "-"))
        logger.info(
            "{n} qualifying ({p:.2f}%)".format(n=q_test.sum(), p=100 * q_test.mean())
        )
        log_classification_metrics(y_true=y_true, y_score=y_score, logger=logger)

        # Additional threshold-free metrics (important under class imbalance):
        # - AUROC is prevalence-insensitive but can be misleading when positives are rare
        # - AUPRC is more informative for rare-event prediction
        # - Calibration metrics ensure probabilistic outputs are interpretable
        try:
            auprc = float(skl.metrics.average_precision_score(y_true=y_true, y_score=y_score))
        except Exception:
            auprc = float("nan")
        try:
            brier = float(skl.metrics.brier_score_loss(y_true=y_true, y_prob=y_score))
        except Exception:
            brier = float("nan")
        try:
            ece = _ece(y_true=y_true, y_prob=y_score, n_bins=int(args.calibration_bins))
        except Exception:
            ece = float("nan")
        logger.info(f"auprc: {auprc:.3f}")
        logger.info(f"brier: {brier:.4f}")
        logger.info(f"ece_{int(args.calibration_bins)}: {ece:.4f}")

        # Thresholded metrics using val-selected threshold. A fixed 0.5 threshold is generally
        # suboptimal unless probabilities are perfectly calibrated AND the loss/cost is symmetric.
        if chosen_threshold is not None:
            y_pred = (y_score >= float(chosen_threshold)).astype(int)
            for met in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
                try:
                    fn = getattr(skl.metrics, f"{met}_score")
                    val = float(fn(y_true=y_true, y_pred=y_pred))
                except Exception:
                    val = float("nan")
                logger.info(f"{met}@thr={chosen_threshold:.4f}: {val:.3f}")

        # Bootstrap confidence intervals (CIs) on the test set for threshold-free metrics.
        if int(args.bootstrap_n) > 0:
            n_boot = int(args.bootstrap_n)
            seed = int(args.bootstrap_seed)
            auroc_ci = _bootstrap_ci(
                y_true,
                y_score,
                metric_fn=lambda yt, ys: skl.metrics.roc_auc_score(y_true=yt, y_score=ys),
                n_boot=n_boot,
                seed=seed,
            )
            auprc_ci = _bootstrap_ci(
                y_true,
                y_score,
                metric_fn=lambda yt, ys: skl.metrics.average_precision_score(y_true=yt, y_score=ys),
                n_boot=n_boot,
                seed=seed + 1,
            )
            brier_ci = _bootstrap_ci(
                y_true,
                y_score,
                metric_fn=lambda yt, ys: skl.metrics.brier_score_loss(y_true=yt, y_prob=ys),
                n_boot=n_boot,
                seed=seed + 2,
            )
            logger.info(f"roc_auc_ci95: [{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}] (bootstrap_n={n_boot})")
            logger.info(f"auprc_ci95: [{auprc_ci[0]:.3f}, {auprc_ci[1]:.3f}] (bootstrap_n={n_boot})")
            logger.info(f"brier_ci95: [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}] (bootstrap_n={n_boot})")

if args.save_preds:
    for v in versions:
        with open(
            data_dirs[v]["test"].joinpath(
                args.classifier + "-preds-" + model_loc.stem + ".pkl"
            ),
            "wb",
        ) as fp:
            pickle.dump(
                {
                    "qualifiers": {
                        outcome: qualifiers[outcome][v]["test"] for outcome in outcomes
                    },
                    "predictions": {outcome: preds[outcome][v] for outcome in outcomes},
                    "labels": {
                        outcome: labels[outcome][v]["test"] for outcome in outcomes
                    },
                    "metadata": {
                        "classifier": args.classifier,
                        "tune_logreg_C": bool(args.tune_logreg_C),
                        "logreg_C_grid": _parse_float_list(args.logreg_C_grid),
                        "threshold_strategy": args.threshold_strategy,
                        "bootstrap_n": int(args.bootstrap_n),
                        "bootstrap_seed": int(args.bootstrap_seed),
                        "calibration_bins": int(args.calibration_bins),
                    },
                },
                fp,
            )
            fix_perms(fp)

logger.info("---fin")
