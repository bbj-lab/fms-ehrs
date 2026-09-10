#!/usr/bin/env python3

"""
Aggregate saved prediction files into manuscript metric and pairwise comparison
tables.

This script upgrades the older one-off plotting helper into a reusable aggregation
layer for benchmark reporting. It supports:

- classification and regression prediction files
- point estimates and bootstrap confidence intervals
- all pairwise paired comparisons within a run family
- two-sided or one-sided paired permutation tests
- Benjamini-Hochberg FDR correction per metric within an invocation
- optional ROC / PR / calibration plots for classification families

The intended use is one invocation per experiment × metric family, which matches
the manuscript-level multiple-testing correction policy.

For a clearer public entrypoint, prefer invoking
`fms_ehrs/scripts/summarize_prediction_family.py`. This module is retained for
backward compatibility with existing workflows.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import math
import pathlib
import pickle
import typing

import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
from sklearn import metrics as skl_mets

from fms_ehrs.framework.artifacts import model_artifact_stem
from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import (
    plot_calibration_curve,
    plot_precision_recall_curve,
    plot_roc_curve,
)

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


Pathlike: typing.TypeAlias = pathlib.PurePath | str | pathlib.Path


class MetricSpec(typing.NamedTuple):
    name: str
    higher_is_better: bool
    fn: typing.Callable[[np.ndarray, np.ndarray], float]


def _ece(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    sort_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sort_idx]
    y_prob_sorted = y_prob[sort_idx]
    bins_true = np.array_split(y_true_sorted, n_bins)
    bins_prob = np.array_split(y_prob_sorted, n_bins)
    total = len(y_true)
    ece = 0.0
    for yt, yp in zip(bins_true, bins_prob):
        if len(yt) == 0:
            continue
        ece += (len(yt) / total) * abs(float(np.mean(yt)) - float(np.mean(yp)))
    return float(ece)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(skl_mets.roc_auc_score(y_true=y_true, y_score=y_score))


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(skl_mets.average_precision_score(y_true=y_true, y_score=y_score))


def _brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(skl_mets.brier_score_loss(y_true=y_true, y_proba=y_score))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(skl_mets.r2_score(y_true=y_true, y_pred=y_pred))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(skl_mets.mean_absolute_error(y_true=y_true, y_pred=y_pred))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(skl_mets.mean_squared_error(y_true=y_true, y_pred=y_pred)))


def _spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(scipy.stats.spearmanr(y_true, y_pred).statistic)


def _classification_metric_specs(ece_bins: int) -> list[MetricSpec]:
    return [
        MetricSpec("roc_auc", True, _roc_auc),
        MetricSpec("pr_auc", True, _pr_auc),
        MetricSpec("brier", False, _brier),
        MetricSpec(
            f"ece_{int(ece_bins)}",
            False,
            lambda yt, ys: _ece(yt, ys, n_bins=int(ece_bins)),
        ),
    ]


def _regression_metric_specs() -> list[MetricSpec]:
    return [
        MetricSpec("spearman_rho", True, _spearman_rho),
        MetricSpec("r2", True, _r2),
        MetricSpec("mae", False, _mae),
        MetricSpec("rmse", False, _rmse),
    ]


def _bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric_fn: typing.Callable[[np.ndarray, np.ndarray], float],
    n_samples: int,
    alpha: float,
    seed: int,
    require_two_classes: bool,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: list[float] = []
    tries = 0
    max_tries = max(n_samples * 20, 100)
    while len(vals) < n_samples and tries < max_tries:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if require_two_classes and np.unique(yt).size < 2:
            continue
        try:
            vals.append(float(metric_fn(yt, ys)))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(vals, alpha / 2)),
        float(np.quantile(vals, 1 - alpha / 2)),
    )


def _paired_bootstrap_diff_ci(
    y_true: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    *,
    metric_fn: typing.Callable[[np.ndarray, np.ndarray], float],
    n_samples: int,
    alpha: float,
    seed: int,
    require_two_classes: bool,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs: list[float] = []
    tries = 0
    max_tries = max(n_samples * 20, 100)
    while len(diffs) < n_samples and tries < max_tries:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys0 = y0[idx]
        ys1 = y1[idx]
        if require_two_classes and np.unique(yt).size < 2:
            continue
        try:
            diffs.append(float(metric_fn(yt, ys1) - metric_fn(yt, ys0)))
        except Exception:
            continue
    if not diffs:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(diffs, alpha / 2)),
        float(np.quantile(diffs, 1 - alpha / 2)),
    )


def _paired_permutation_pval(
    y_true: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    *,
    metric_fn: typing.Callable[[np.ndarray, np.ndarray], float],
    higher_is_better: bool,
    n_samples: int,
    seed: int,
    alternative: typing.Literal["one-sided", "two-sided"],
    require_two_classes: bool,
) -> float:
    rng = np.random.default_rng(seed)
    obs_raw = float(metric_fn(y_true, y1) - metric_fn(y_true, y0))
    obs = obs_raw if higher_is_better else -obs_raw
    count = 0
    valid = 0
    n = len(y_true)
    for _ in range(n_samples):
        swap = rng.integers(0, 2, size=n).astype(bool)
        perm0 = np.where(swap, y1, y0)
        perm1 = np.where(swap, y0, y1)
        if require_two_classes and np.unique(y_true).size < 2:
            continue
        try:
            diff_raw = float(metric_fn(y_true, perm1) - metric_fn(y_true, perm0))
        except Exception:
            continue
        valid += 1
        if alternative == "two-sided":
            count += abs(diff_raw) >= abs(obs_raw)
        else:
            diff = diff_raw if higher_is_better else -diff_raw
            count += diff >= obs
    if valid == 0:
        return float("nan")
    return float((count + 1) / (valid + 1))


def _bh_adjust(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (math.isnan(pvals[i]), pvals[i]))
    adj = [float("nan")] * n
    prev = 1.0
    for rank_rev, idx in enumerate(reversed(order), start=1):
        p = pvals[idx]
        if math.isnan(p):
            adj[idx] = float("nan")
            continue
        rank = n - rank_rev + 1
        val = min(prev, p * n / rank)
        adj[idx] = min(val, 1.0)
        prev = val
    return adj


def _write_df(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        cols = list(df.columns)
        header = "| " + " | ".join(map(str, cols)) + " |\n"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(map(lambda x: "" if pd.isna(x) else str(x), row.tolist())) + " |\n")
        path.write_text(header + sep + "".join(rows))


def _load_payload(path: pathlib.Path) -> dict:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def _resolve_outcomes_path(
    pred_path: pathlib.Path,
    payload: dict,
    explicit_outcomes_path: Pathlike | None = None,
) -> pathlib.Path | None:
    if explicit_outcomes_path is not None:
        explicit_path = pathlib.Path(explicit_outcomes_path).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = pred_path.parent.joinpath(explicit_path)
        outcomes_path = explicit_path.resolve()
        if not outcomes_path.exists():
            raise FileNotFoundError(
                f"Explicit outcomes parquet {explicit_outcomes_path!r} for prediction payload "
                f"{pred_path} does not exist at {outcomes_path}."
            )
        return outcomes_path
    metadata = payload.get("metadata", {}) or {}
    outcomes_name = metadata.get("outcomes_parquet")
    if not outcomes_name:
        return None
    outcomes_path = pred_path.parent.joinpath(str(outcomes_name)).expanduser().resolve()
    if not outcomes_path.exists():
        raise FileNotFoundError(
            f"Prediction payload at {pred_path} points to outcomes parquet "
            f"{outcomes_name!r}, but {outcomes_path} does not exist."
        )
    return outcomes_path


def _load_hospitalization_ids(
    outcomes_path: pathlib.Path,
    *,
    id_cache: dict[pathlib.Path, np.ndarray],
) -> np.ndarray:
    if outcomes_path in id_cache:
        return id_cache[outcomes_path]

    outcomes_scan = pl.scan_parquet(outcomes_path)
    schema = outcomes_scan.collect_schema()
    names = schema.names()
    if "hospitalization_id" in names:
        expr = pl.col("hospitalization_id").cast(pl.String)
    elif "hadm_id" in names:
        expr = pl.col("hadm_id").cast(pl.String)
    else:
        raise ValueError(
            f"Could not find hospitalization_id or hadm_id in outcomes parquet {outcomes_path}."
        )

    ids = (
        outcomes_scan
        .select(expr.alias("hospitalization_id"))
        .collect()
        .to_series()
        .to_numpy()
        .astype(str)
    )
    id_cache[outcomes_path] = ids
    return ids


def _derive_pred_paths(
    *,
    data_dir: pathlib.Path,
    data_versions: list[str],
    classifier: str,
    task_type: str,
    model_loc: pathlib.Path,
) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    # Only the run's own stem is accepted. Resolved-checkpoint fallbacks are
    # ambiguous whenever several runs publish the same checkpoint basename.
    model_stems = [model_artifact_stem(model_loc)]
    for dv in data_versions:
        test_dir = data_dir.joinpath(f"{dv}-tokenized", "test")
        classifier_prefixes = [classifier]
        if task_type == "regression" and not classifier.startswith("reg_"):
            classifier_prefixes = [f"reg_{classifier}", classifier]

        matches: list[pathlib.Path] = []
        for prefix in classifier_prefixes:
            for model_stem in model_stems:
                legacy = test_dir.joinpath(f"{prefix}-preds-{model_stem}.pkl")
                if legacy.exists():
                    matches.append(legacy)
                matches.extend(sorted(test_dir.glob(f"{prefix}-preds-*-{model_stem}.pkl")))

        deduped_matches = list(dict.fromkeys(matches))
        if len(deduped_matches) == 1:
            paths.append(deduped_matches[0])
            continue
        if len(deduped_matches) == 0:
            raise FileNotFoundError(
                f"No prediction pickle found for data_version={dv!r}, classifier={classifier!r}, "
                f"model={model_artifact_stem(model_loc)!r} under {test_dir}. "
                "Provide --pred_paths explicitly if you are using a nonstandard layout."
            )
        raise ValueError(
            f"Multiple tagged prediction pickles match data_version={dv!r}, classifier={classifier!r}, "
                f"model={model_artifact_stem(model_loc)!r} under {test_dir}: {[p.name for p in deduped_matches]}. "
            "Provide --pred_paths explicitly."
        )
    return paths


def _load_named_results(
    *,
    pred_paths: list[pathlib.Path],
    handles: list[str],
    outcomes_paths: list[Pathlike | None] | None = None,
) -> collections.OrderedDict[str, dict]:
    named = collections.OrderedDict()
    if outcomes_paths is None:
        outcomes_paths = [None] * len(pred_paths)
    for handle, path, outcomes_path in zip(handles, pred_paths, outcomes_paths):
        payload = _load_payload(path)
        named[handle] = {
            "payload": payload,
            "pred_path": path,
            "explicit_outcomes_path": outcomes_path,
        }
    return named


def _available_outcomes(named_results: collections.OrderedDict[str, dict]) -> list[str]:
    common: set[str] | None = None
    for result in named_results.values():
        preds = set(result["payload"]["predictions"].keys())
        common = preds if common is None else common & preds
    return sorted(common or [])


def _extract_arrays(
    payload: dict,
    outcome: str,
    *,
    pred_path: pathlib.Path,
    id_cache: dict[pathlib.Path, np.ndarray],
    explicit_outcomes_path: Pathlike | None = None,
) -> dict[str, typing.Any]:
    quals = np.asarray(payload["qualifiers"][outcome]).astype(bool)
    labels = np.asarray(payload["labels"][outcome]).astype(float)
    y_true = labels[quals].astype(float)
    y_pred = np.asarray(payload["predictions"][outcome]).astype(float)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Outcome {outcome!r}: label/pred length mismatch ({y_true.shape[0]} vs {y_pred.shape[0]})"
        )
    outcomes_path = _resolve_outcomes_path(
        pred_path,
        payload,
        explicit_outcomes_path=explicit_outcomes_path,
    )
    ids = None
    if outcomes_path is not None:
        full_ids = _load_hospitalization_ids(outcomes_path, id_cache=id_cache)
        if full_ids.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Outcome {outcome!r}: hospitalization_id length mismatch for {outcomes_path} "
                f"({full_ids.shape[0]} ids vs {labels.shape[0]} labels)."
            )
        ids = full_ids[quals]
        if ids.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Outcome {outcome!r}: hospitalization_id/pred length mismatch "
                f"({ids.shape[0]} vs {y_pred.shape[0]})."
            )

    return {
        "y_true": y_true,
        "y_score": y_pred,
        "hospitalization_ids": ids,
        "outcomes_parquet": str(outcomes_path) if outcomes_path is not None else None,
    }


def _align_pairwise_arrays(
    *,
    arr0: dict[str, typing.Any],
    arr1: dict[str, typing.Any],
    outcome: str,
    handle0: str,
    handle1: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    ids0 = arr0.get("hospitalization_ids")
    ids1 = arr1.get("hospitalization_ids")
    if ids0 is not None and ids1 is not None:
        df0 = pd.DataFrame(
            {
                "hospitalization_id": np.asarray(ids0).astype(str),
                "y_true_0": np.asarray(arr0["y_true"]).astype(float),
                "y_score_0": np.asarray(arr0["y_score"]).astype(float),
            }
        )
        df1 = pd.DataFrame(
            {
                "hospitalization_id": np.asarray(ids1).astype(str),
                "y_true_1": np.asarray(arr1["y_true"]).astype(float),
                "y_score_1": np.asarray(arr1["y_score"]).astype(float),
            }
        )
        if df0["hospitalization_id"].duplicated().any():
            raise ValueError(
                f"Outcome {outcome!r}: duplicate hospitalization_id values for handle {handle0!r}."
            )
        if df1["hospitalization_id"].duplicated().any():
            raise ValueError(
                f"Outcome {outcome!r}: duplicate hospitalization_id values for handle {handle1!r}."
            )
        merged = df0.merge(
            df1,
            on="hospitalization_id",
            how="inner",
            sort=True,
            validate="1:1",
        )
        if merged.empty:
            raise ValueError(
                f"Outcome {outcome!r}: no overlapping hospitalization_id values between "
                f"{handle0!r} and {handle1!r}."
            )
        y_true0 = merged["y_true_0"].to_numpy(dtype=float)
        y_true1 = merged["y_true_1"].to_numpy(dtype=float)
        if not np.allclose(y_true0, y_true1, equal_nan=True):
            raise ValueError(
                f"Outcome {outcome!r}: labels differ after hospitalization_id alignment "
                f"between {handle0!r} and {handle1!r}."
            )
        return (
            y_true0,
            merged["y_score_0"].to_numpy(dtype=float),
            merged["y_score_1"].to_numpy(dtype=float),
            int(merged.shape[0]),
            "hospitalization_id",
        )

    y_true0 = np.asarray(arr0["y_true"]).astype(float)
    y_true1 = np.asarray(arr1["y_true"]).astype(float)
    if y_true0.shape[0] != y_true1.shape[0] or not np.array_equal(y_true0, y_true1):
        raise ValueError(
            f"Outcome {outcome!r}: cannot do paired comparison because labels differ "
            f"between {handle0!r} and {handle1!r}."
        )
    return (
        y_true0,
        np.asarray(arr0["y_score"]).astype(float),
        np.asarray(arr1["y_score"]).astype(float),
        int(y_true0.shape[0]),
        "row_order",
    )


def _metric_specs(task_type: str, ece_bins: int) -> list[MetricSpec]:
    if task_type == "classification":
        return _classification_metric_specs(ece_bins)
    return _regression_metric_specs()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family_name", type=str, default="family")
    parser.add_argument("--task_type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--pred_paths", type=pathlib.Path, nargs="*", default=None)
    parser.add_argument("--handles", type=str, nargs="*", default=None)
    parser.add_argument("--outcomes_paths", type=str, nargs="*", default=None)
    parser.add_argument("--baseline_handle", type=str, default=None)
    parser.add_argument("--outcomes", type=str, nargs="*", default=None)
    parser.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("./aggregation"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap_n", type=int, default=2000)
    parser.add_argument(
        "--pairwise_bootstrap_n",
        type=int,
        default=0,
        help="Bootstrap samples for pairwise delta CIs. Defaults to 0 because family-level summaries only need metric CIs.",
    )
    parser.add_argument("--permutation_n", type=int, default=2000)
    parser.add_argument("--ece_bins", type=int, default=15)
    parser.add_argument("--alternative", choices=["one-sided", "two-sided"], default="two-sided")
    parser.add_argument("--fdr", choices=["bh", "none"], default="bh")
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--plots_dir", type=pathlib.Path, default=None)

    # Backward-compatible path-derivation mode
    parser.add_argument("--data_dir", type=pathlib.Path, default=None)
    parser.add_argument("--data_versions", type=str, nargs="*", default=None)
    parser.add_argument("--classifier", choices=["light_gbm", "logistic_regression_cv", "logistic_regression", "mlp", "ridge_regression"], default=None)
    parser.add_argument("--model_loc", type=pathlib.Path, default=None)

    args, _ = parser.parse_known_args()
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    if args.pred_paths:
        pred_paths = [pathlib.Path(p).expanduser().resolve() for p in args.pred_paths]
        if not args.handles or len(args.handles) != len(pred_paths):
            raise ValueError("--handles must be provided and match --pred_paths length.")
        handles = list(args.handles)
        if args.outcomes_paths is not None and len(args.outcomes_paths) != len(pred_paths):
            raise ValueError("--outcomes_paths must match --pred_paths length when provided.")
        outcomes_paths = list(args.outcomes_paths) if args.outcomes_paths is not None else [None] * len(pred_paths)
    else:
        if not (args.data_dir and args.data_versions and args.classifier and args.model_loc and args.handles):
            raise ValueError(
                "Provide either --pred_paths/--handles or the backward-compatible "
                "--data_dir/--data_versions/--classifier/--model_loc/--handles set."
            )
        data_dir = pathlib.Path(args.data_dir).expanduser().resolve()
        model_loc = pathlib.Path(args.model_loc).expanduser()
        pred_paths = _derive_pred_paths(
            data_dir=data_dir,
            data_versions=list(args.data_versions),
            classifier=str(args.classifier),
            task_type=str(args.task_type),
            model_loc=model_loc,
        )
        handles = list(args.handles)
        if len(handles) != len(pred_paths):
            raise ValueError("--handles must match the number of derived prediction paths.")
        if args.outcomes_paths is not None:
            raise ValueError("--outcomes_paths is only supported with explicit --pred_paths.")
        outcomes_paths = [None] * len(pred_paths)

    paired_inputs = list(zip(handles, pred_paths, outcomes_paths))
    baseline = None
    if args.baseline_handle is not None:
        baseline = str(args.baseline_handle)
        if baseline not in handles:
            raise ValueError(f"--baseline_handle={baseline!r} is not present in --handles.")
        paired_inputs = [pair for pair in paired_inputs if pair[0] == baseline] + [
            pair for pair in paired_inputs if pair[0] != baseline
        ]
    handles = [h for h, _, _ in paired_inputs]
    pred_paths = [p for _, p, _ in paired_inputs]
    outcomes_paths = [o for _, _, o in paired_inputs]

    named_results = _load_named_results(
        pred_paths=pred_paths,
        handles=handles,
        outcomes_paths=outcomes_paths,
    )
    outcomes = list(args.outcomes) if args.outcomes else _available_outcomes(named_results)
    metric_specs = _metric_specs(args.task_type, args.ece_bins)
    plots_dir = pathlib.Path(args.plots_dir).expanduser().resolve() if args.plots_dir else pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    id_cache: dict[pathlib.Path, np.ndarray] = {}

    metrics_rows: list[dict[str, typing.Any]] = []
    pairwise_rows: list[dict[str, typing.Any]] = []

    for outcome in outcomes:
        logger.info(outcome.upper().ljust(79, "-"))
        named_arrays: collections.OrderedDict[str, dict[str, typing.Any]] = collections.OrderedDict()
        for handle, result in named_results.items():
            payload = result["payload"]
            if outcome not in payload["predictions"]:
                logger.info(f"Skipping outcome={outcome} for handle={handle}: not present in predictions.")
                continue
            arrays = _extract_arrays(
                payload,
                outcome,
                pred_path=result["pred_path"],
                id_cache=id_cache,
                explicit_outcomes_path=result.get("explicit_outcomes_path"),
            )
            y_true = np.asarray(arrays["y_true"]).astype(float)
            y_pred = np.asarray(arrays["y_score"]).astype(float)
            named_arrays[handle] = arrays
            require_two_classes = args.task_type == "classification"
            for spec in metric_specs:
                try:
                    point = float(spec.fn(y_true, y_pred))
                except Exception:
                    point = float("nan")
                ci_lo, ci_hi = _bootstrap_metric_ci(
                    y_true,
                    y_pred,
                    metric_fn=spec.fn,
                    n_samples=int(args.bootstrap_n),
                    alpha=float(args.alpha),
                    seed=42,
                    require_two_classes=require_two_classes,
                )
                metrics_rows.append(
                    {
                        "family_name": args.family_name,
                        "task_type": args.task_type,
                        "outcome": outcome,
                        "handle": handle,
                        "metric": spec.name,
                        "higher_is_better": spec.higher_is_better,
                        "n_eval": int(y_true.shape[0]),
                        "point": point,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "pred_path": str(result["pred_path"]),
                        "outcomes_parquet": arrays["outcomes_parquet"],
                    }
                )

        if args.make_plots and args.task_type == "classification" and len(named_arrays) >= 2:
            plot_calibration_curve(
                named_arrays,
                n_bins=int(args.ece_bins),
                savepath=plots_dir.joinpath(f"cal-{args.family_name}-{outcome}.pdf"),
            )
            plot_roc_curve(
                named_arrays,
                savepath=plots_dir.joinpath(f"roc-{args.family_name}-{outcome}.pdf"),
            )
            plot_precision_recall_curve(
                named_arrays,
                savepath=plots_dir.joinpath(f"pr-{args.family_name}-{outcome}.pdf"),
            )

        if baseline is not None and baseline in named_arrays:
            pair_iter = ((baseline, handle1) for handle1 in named_arrays.keys() if handle1 != baseline)
        else:
            pair_iter = itertools.combinations(named_arrays.keys(), 2)
        for handle0, handle1 in pair_iter:
            y_true, y0, y1, n_eval_common, alignment_mode = _align_pairwise_arrays(
                arr0=named_arrays[handle0],
                arr1=named_arrays[handle1],
                outcome=outcome,
                handle0=handle0,
                handle1=handle1,
            )
            require_two_classes = args.task_type == "classification"
            for spec in metric_specs:
                try:
                    metric0 = float(spec.fn(y_true, y0))
                    metric1 = float(spec.fn(y_true, y1))
                    delta = metric1 - metric0
                except Exception:
                    metric0 = float("nan")
                    metric1 = float("nan")
                    delta = float("nan")
                if int(args.pairwise_bootstrap_n) > 0:
                    ci_lo, ci_hi = _paired_bootstrap_diff_ci(
                        y_true,
                        y0,
                        y1,
                        metric_fn=spec.fn,
                        n_samples=int(args.pairwise_bootstrap_n),
                        alpha=float(args.alpha),
                        seed=123,
                        require_two_classes=require_two_classes,
                    )
                else:
                    ci_lo, ci_hi = float("nan"), float("nan")
                p_raw = _paired_permutation_pval(
                    y_true,
                    y0,
                    y1,
                    metric_fn=spec.fn,
                    higher_is_better=spec.higher_is_better,
                    n_samples=int(args.permutation_n),
                    seed=999,
                    alternative=typing.cast(typing.Literal["one-sided", "two-sided"], args.alternative),
                    require_two_classes=require_two_classes,
                )
                delta_better = delta if spec.higher_is_better else -delta
                pairwise_rows.append(
                    {
                        "family_name": args.family_name,
                        "task_type": args.task_type,
                        "outcome": outcome,
                        "metric": spec.name,
                        "higher_is_better": spec.higher_is_better,
                        "handle0": handle0,
                        "handle1": handle1,
                        "n_eval_common": n_eval_common,
                        "alignment_mode": alignment_mode,
                        "metric0": metric0,
                        "metric1": metric1,
                        "delta_raw": delta,
                        "delta_better": delta_better,
                        "delta_ci_lo_raw": ci_lo,
                        "delta_ci_hi_raw": ci_hi,
                        "delta_ci_lo_better": ci_lo if spec.higher_is_better else -ci_hi,
                        "delta_ci_hi_better": ci_hi if spec.higher_is_better else -ci_lo,
                        "p_raw": p_raw,
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df["p_adj"] = np.nan
        if args.fdr == "bh":
            for metric, idxs in pairwise_df.groupby("metric").groups.items():
                pvals = pairwise_df.loc[list(idxs), "p_raw"].astype(float).tolist()
                pairwise_df.loc[list(idxs), "p_adj"] = _bh_adjust(pvals)
        else:
            pairwise_df["p_adj"] = pairwise_df["p_raw"]

    metrics_path = out_dir.joinpath(f"{args.family_name}-metrics.csv")
    pairwise_path = out_dir.joinpath(f"{args.family_name}-pairwise.csv")
    _write_df(metrics_df, metrics_path)
    _write_df(pairwise_df, pairwise_path)

    pretty_metrics = metrics_df.copy()
    if not pretty_metrics.empty:
        pretty_metrics["estimate_ci95"] = pretty_metrics.apply(
            lambda r: f"{r['point']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
            if pd.notnull(r["point"]) else "nan",
            axis=1,
        )
        _write_df(
            pretty_metrics[
                ["family_name", "task_type", "outcome", "handle", "metric", "n_eval", "estimate_ci95"]
            ],
            out_dir.joinpath(f"{args.family_name}-metrics.md"),
        )

    pretty_pairwise = pairwise_df.copy()
    if not pretty_pairwise.empty:
        pretty_pairwise["delta_better_ci95"] = pretty_pairwise.apply(
            lambda r: f"{r['delta_better']:.4f} [{r['delta_ci_lo_better']:.4f}, {r['delta_ci_hi_better']:.4f}]"
            if pd.notnull(r["delta_better"]) else "nan",
            axis=1,
        )
        _write_df(
            pretty_pairwise[
                [
                    "family_name",
                    "task_type",
                    "outcome",
                    "metric",
                    "handle0",
                    "handle1",
                    "n_eval_common",
                    "alignment_mode",
                    "delta_better_ci95",
                    "p_raw",
                    "p_adj",
                ]
            ],
            out_dir.joinpath(f"{args.family_name}-pairwise.md"),
        )

    logger.info(f"Wrote metrics table: {metrics_path}")
    logger.info(f"Wrote pairwise table: {pairwise_path}")
    logger.info("---fin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
