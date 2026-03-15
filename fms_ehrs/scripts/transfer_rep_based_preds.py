#!/usr/bin/env python3

"""
make some simple predictions outcomes ~ features
provide some performance breakdowns

Supports:
  - Classification: logistic_regression, logistic_regression_cv, mlp, light_gbm
  - Regression: ridge_regression (with R², MAE, RMSE, Spearman ρ)
"""

import argparse
import collections
import pathlib
import pickle
import re
import typing

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.stats
import sklearn as skl
import sklearn.neural_network
import sklearn.linear_model

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
    """Quantile-binned Expected calibration error (ECE) for probabilistic binary predictions."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    
    # 1. Sort by predicted probability
    sort_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sort_idx]
    y_prob_sorted = y_prob[sort_idx]
    
    # 2. Split into n_bins of equal size
    bins_true = np.array_split(y_true_sorted, n_bins)
    bins_prob = np.array_split(y_prob_sorted, n_bins)
    
    ece = 0.0
    total_samples = len(y_true)
    
    for tr, pr in zip(bins_true, bins_prob):
        if len(tr) == 0:
            continue
        acc = float(np.mean(tr))
        conf = float(np.mean(pr))
        weight = len(tr) / total_samples
        ece += weight * abs(acc - conf)
        
    return float(ece)


def _sanitize_preds_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip()).strip("-")


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
    require_two_classes: bool = True,
) -> tuple[float, float]:
    """Bootstrap CI for metrics that take (y_true, y_score)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = y_true.shape[0]
    vals: list[float] = []
    tries = 0
    while len(vals) < n_boot and tries < (n_boot * 10):
        tries += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        # Some metrics (e.g., AUROC) are undefined if a resample has only one class.
        if require_two_classes and np.unique(yt).size < 2:
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
    choices=["light_gbm", "logistic_regression_cv", "logistic_regression", "mlp", "ridge_regression"],
    default="logistic_regression",
)
parser.add_argument(
    "--task_type",
    choices=["classification", "regression"],
    default="classification",
    help="Task type: classification (AUROC etc.) or regression (R², MAE, RMSE, Spearman).",
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
    "--outcomes_parquet",
    type=str,
    default="tokens_timelines_outcomes.parquet",
    help="Parquet filename containing outcome labels (default: tokens_timelines_outcomes.parquet).",
)
parser.add_argument(
    "--preds_tag",
    type=str,
    default="",
    help="Optional tag to encode the outcome family / label source in saved prediction filenames.",
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
    "--ridge_alpha_grid",
    type=str,
    default="[0.01,0.1,1,10,100]",
    help="Grid of alpha values for Ridge regression val tuning.",
)
parser.add_argument(
    "--mlp_hidden_sizes",
    type=str,
    default="256",
    help="Comma-separated hidden layer sizes for MLP classifier (e.g., '256' or '512,256').",
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
is_regression = args.task_type == "regression"

if args.outcomes is None:
    if is_regression:
        outcomes = ("length_of_stay", "icu_length_of_stay")
    else:
        outcomes = ("same_admission_death", "long_length_of_stay", "imv_event") + (
            ("icu_admission",) if not args.drop_icu_adm else ()
        )
else:
    outcomes = tuple(str(x) for x in args.outcomes if str(x).strip())
    if args.drop_icu_adm:
        outcomes = tuple(o for o in outcomes if o != "icu_admission")
    if not outcomes:
        raise ValueError("No outcomes specified (empty --outcomes after filtering).")

outcomes_parquet = args.outcomes_parquet

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
        outcomes_scan = pl.scan_parquet(data_dirs[v][s].joinpath(outcomes_parquet))
        outcomes_schema = outcomes_scan.collect_schema()
        for outcome in outcomes:
            raw_labels = (
                outcomes_scan
                .select(pl.col(outcome).cast(pl.Float64))
                .collect()
                .to_numpy()
                .ravel()
                .astype(float)
            )
            labels[outcome][v][s] = raw_labels
            if is_regression:
                # For regression: valid = non-NaN values (some admissions may lack the target)
                qualifiers[outcome][v][s] = np.isfinite(raw_labels)
            else:
                qualifiers[outcome][v][s] = np.isfinite(raw_labels)
                # Exclude admissions that already met the outcome during the first 24h
                # whenever a parallel <outcome>_24h column exists in the outcomes parquet.
                outcome_24h = outcome + "_24h"
                if outcome_24h in outcomes_schema:
                    qualifiers[outcome][v][s] &= ~(
                        outcomes_scan
                        .select(pl.col(outcome_24h).fill_null(False))
                        .collect()
                        .to_numpy()
                        .ravel()
                        .astype(bool)
                    )


preds = collections.defaultdict(dict)
skipped_outcomes: set[str] = set()

for outcome in outcomes:
    logger.info(outcome.replace("_", " ").upper().ljust(79, "-"))

    Xtrain = (features["orig"]["train"])[qualifiers[outcome]["orig"]["train"]]
    ytrain = (labels[outcome]["orig"]["train"])[qualifiers[outcome]["orig"]["train"]]
    Xval = (features["orig"]["val"])[qualifiers[outcome]["orig"]["val"]]
    yval = (labels[outcome]["orig"]["val"])[qualifiers[outcome]["orig"]["val"]]

    logger.info(f"train: {Xtrain.shape[0]}, val: {Xval.shape[0]}")
    if Xtrain.shape[0] == 0 or Xval.shape[0] == 0 or ytrain.size == 0 or yval.size == 0:
        logger.info(
            f"SKIPPING {outcome}: empty qualifying train/val set "
            f"(train={Xtrain.shape[0]}, val={Xval.shape[0]})."
        )
        skipped_outcomes.add(outcome)
        continue
    if any(qualifiers[outcome][v]["test"].sum() == 0 for v in versions):
        logger.info(
            f"SKIPPING {outcome}: empty qualifying test set for at least one version "
            f"(orig={qualifiers[outcome]['orig']['test'].sum()}, new={qualifiers[outcome]['new']['test'].sum()})."
        )
        skipped_outcomes.add(outcome)
        continue

    # Guard: skip outcomes where training or val data has fewer than 2 classes
    # (e.g., icu_admission in ICU-only cohorts where all patients are positive).
    if not is_regression:
        n_train_classes = len(np.unique(ytrain[np.isfinite(ytrain)]))
        n_val_classes = len(np.unique(yval[np.isfinite(yval)]))
        if n_train_classes < 2 or n_val_classes < 2:
            logger.info(
                f"SKIPPING {outcome}: only {n_train_classes} class(es) in train, "
                f"{n_val_classes} in val (need >=2 for classification)."
            )
            skipped_outcomes.add(outcome)
            continue
    if is_regression:
        logger.info(f"  ytrain range: [{ytrain.min():.2f}, {ytrain.max():.2f}], mean={ytrain.mean():.2f}, std={ytrain.std():.2f}")

    # ---------------------------------------------------------------
    # Fit estimator
    # ---------------------------------------------------------------
    if is_regression:
        match args.classifier:
            case "ridge_regression":
                alpha_grid = _parse_float_list(args.ridge_alpha_grid)
                if not alpha_grid:
                    alpha_grid = [1.0]

                def _fit_ridge(alpha: float):
                    est = skl.pipeline.make_pipeline(
                        skl.preprocessing.StandardScaler(),
                        skl.compose.TransformedTargetRegressor(
                            regressor=skl.linear_model.Ridge(
                                alpha=float(alpha),
                                random_state=42,
                            ),
                            transformer=skl.preprocessing.StandardScaler(),
                        )
                    )
                    est.fit(X=Xtrain, y=ytrain)
                    return est

                best_alpha = 1.0
                if len(alpha_grid) > 1:
                    best_val = -float("inf")
                    for alpha in alpha_grid:
                        est = _fit_ridge(alpha)
                        y_pred_val = est.predict(Xval)
                        try:
                            r2 = float(skl.metrics.r2_score(y_true=yval, y_pred=y_pred_val))
                        except Exception:
                            r2 = -float("inf")
                        if r2 > best_val:
                            best_val = r2
                            best_alpha = float(alpha)
                    logger.info(f"Selected Ridge alpha on val (R²): alpha={best_alpha} (grid={alpha_grid})")
                else:
                    logger.info(f"Using Ridge alpha={best_alpha} (no tuning; grid={alpha_grid})")

                estimator = _fit_ridge(best_alpha)

            case "mlp":
                hidden_sizes = tuple(int(x) for x in args.mlp_hidden_sizes.split(",") if x.strip())
                estimator = skl.pipeline.make_pipeline(
                    skl.preprocessing.StandardScaler(),
                    skl.neural_network.MLPRegressor(
                        hidden_layer_sizes=hidden_sizes,
                        activation="relu",
                        max_iter=500,
                        early_stopping=False,
                        validation_fraction=0.1,
                        random_state=42,
                    ),
                )
                estimator.fit(X=Xtrain, y=ytrain)

            case _:
                # Fall back to Ridge for unrecognized classifiers in regression mode
                logger.info(f"Classifier '{args.classifier}' not suitable for regression; falling back to ridge_regression.")
                estimator = skl.pipeline.make_pipeline(
                    skl.preprocessing.StandardScaler(),
                    skl.linear_model.Ridge(alpha=1.0, random_state=42),
                )
                estimator.fit(X=Xtrain, y=ytrain)

    else:
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

            case "mlp":
                hidden_sizes = tuple(int(x) for x in args.mlp_hidden_sizes.split(",") if x.strip())
                estimator = skl.pipeline.make_pipeline(
                    skl.preprocessing.StandardScaler(),
                    skl.neural_network.MLPClassifier(
                        hidden_layer_sizes=hidden_sizes,
                        activation="relu",
                        max_iter=500,
                        early_stopping=False,
                        validation_fraction=0.1,
                        random_state=42,
                    ),
                )
                estimator.fit(X=Xtrain, y=ytrain)

            case _:
                raise NotImplementedError(
                    f"Classifier {args.classifier} is not yet supported."
                )

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------
    if is_regression:
        # ----- Regression evaluation -----
        for v in versions:
            logger.info(v.upper())

            q_test = qualifiers[outcome][v]["test"]
            y_pred = estimator.predict((features[v]["test"])[q_test])
            y_true = (labels[outcome][v]["test"])[q_test]
            preds[outcome][v] = y_pred

            logger.info("REGRESSION PERFORMANCE".ljust(49, "-"))
            logger.info(f"{q_test.sum()} qualifying ({100 * q_test.mean():.2f}%)")

            r2 = float(skl.metrics.r2_score(y_true=y_true, y_pred=y_pred))
            mae = float(skl.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred))
            rmse = float(np.sqrt(skl.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)))
            try:
                spearman_r, spearman_p = scipy.stats.spearmanr(y_true, y_pred)
                spearman_r = float(spearman_r)
            except Exception:
                spearman_r = float("nan")
                spearman_p = float("nan")

            logger.info(f"r2: {r2:.4f}")
            logger.info(f"mae: {mae:.4f}")
            logger.info(f"rmse: {rmse:.4f}")
            logger.info(f"spearman_r: {spearman_r:.4f}")
            logger.info(f"spearman_p: {spearman_p:.2e}")

            # Bootstrap CIs for regression metrics
            if int(args.bootstrap_n) > 0:
                n_boot = int(args.bootstrap_n)
                seed = int(args.bootstrap_seed)
                r2_ci = _bootstrap_ci(
                    y_true, y_pred,
                    metric_fn=lambda yt, ys: skl.metrics.r2_score(y_true=yt, y_pred=ys),
                    n_boot=n_boot, seed=seed, require_two_classes=False,
                )
                mae_ci = _bootstrap_ci(
                    y_true, y_pred,
                    metric_fn=lambda yt, ys: skl.metrics.mean_absolute_error(y_true=yt, y_pred=ys),
                    n_boot=n_boot, seed=seed + 1, require_two_classes=False,
                )
                spearman_ci = _bootstrap_ci(
                    y_true, y_pred,
                    metric_fn=lambda yt, ys: float(scipy.stats.spearmanr(yt, ys).statistic),
                    n_boot=n_boot, seed=seed + 2, require_two_classes=False,
                )
                logger.info(f"r2_ci95: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}] (bootstrap_n={n_boot})")
                logger.info(f"mae_ci95: [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}] (bootstrap_n={n_boot})")
                logger.info(f"spearman_ci95: [{spearman_ci[0]:.4f}, {spearman_ci[1]:.4f}] (bootstrap_n={n_boot})")

    else:
        # ----- Classification evaluation -----
        # Select decision threshold using validation predictions (for probability-based classifiers).
        chosen_threshold = None
        if args.classifier in ("logistic_regression", "logistic_regression_cv", "mlp"):
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
        prefix = args.classifier
        if is_regression:
            prefix = f"reg_{prefix}"
        filename = prefix + "-preds-" + model_loc.stem + ".pkl"
        if args.preds_tag.strip():
            filename = (
                prefix
                + "-preds-"
                + _sanitize_preds_tag(args.preds_tag)
                + "-"
                + model_loc.stem
                + ".pkl"
            )
        with open(
            data_dirs[v]["test"].joinpath(filename),
            "wb",
        ) as fp:
            pickle.dump(
                {
                    "qualifiers": {
                        outcome: qualifiers[outcome][v]["test"] for outcome in outcomes if outcome not in skipped_outcomes
                    },
                    "predictions": {outcome: preds[outcome][v] for outcome in outcomes if outcome not in skipped_outcomes},
                    "labels": {
                        outcome: labels[outcome][v]["test"] for outcome in outcomes if outcome not in skipped_outcomes
                    },
                    "metadata": {
                        "classifier": args.classifier,
                        "task_type": args.task_type,
                        "preds_tag": args.preds_tag,
                        "outcomes_parquet": args.outcomes_parquet,
                        "outcomes": list(outcomes),
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
