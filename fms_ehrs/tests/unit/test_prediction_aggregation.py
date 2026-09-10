from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from fms_ehrs.scripts.aggregate_version_preds import (
    _derive_pred_paths,
    _paired_permutation_pval,
    _rmse,
    main as aggregate_main,
)


class TestPredictionAggregation(unittest.TestCase):
    def test_aggregate_version_preds_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred0 = tmp / "model0.pkl"
            pred1 = tmp / "model1.pkl"
            out_dir = tmp / "out"

            payload0 = {
                "qualifiers": {"outcome_a": np.array([True, True, True, True])},
                "labels": {"outcome_a": np.array([0.0, 1.0, 0.0, 1.0])},
                "predictions": {"outcome_a": np.array([0.1, 0.8, 0.2, 0.7])},
            }
            payload1 = {
                "qualifiers": {"outcome_a": np.array([True, True, True, True])},
                "labels": {"outcome_a": np.array([0.0, 1.0, 0.0, 1.0])},
                "predictions": {"outcome_a": np.array([0.2, 0.7, 0.3, 0.6])},
            }
            pred0.write_bytes(pickle.dumps(payload0))
            pred1.write_bytes(pickle.dumps(payload1))

            argv_old = sys.argv[:]
            try:
                sys.argv = [
                    "aggregate_version_preds.py",
                    "--family_name", "smoke",
                    "--task_type", "classification",
                    "--pred_paths", str(pred0), str(pred1),
                    "--handles", "baseline", "candidate",
                    "--outcomes", "outcome_a",
                    "--out_dir", str(out_dir),
                    "--bootstrap_n", "5",
                    "--permutation_n", "5",
                    "--alternative", "two-sided",
                ]
                rc = aggregate_main()
            finally:
                sys.argv = argv_old

            self.assertEqual(rc, 0)
            metrics_csv = out_dir / "smoke-metrics.csv"
            pairwise_csv = out_dir / "smoke-pairwise.csv"
            metrics_md = out_dir / "smoke-metrics.md"
            pairwise_md = out_dir / "smoke-pairwise.md"

            self.assertTrue(metrics_csv.exists())
            self.assertTrue(pairwise_csv.exists())
            self.assertTrue(metrics_md.exists())
            self.assertTrue(pairwise_md.exists())

            metrics_df = pd.read_csv(metrics_csv)
            pairwise_df = pd.read_csv(pairwise_csv)

            self.assertIn("roc_auc", set(metrics_df["metric"]))
            self.assertIn("pr_auc", set(metrics_df["metric"]))
            self.assertFalse(metrics_df["ci_lo"].isna().all())
            self.assertIn("p_raw", pairwise_df.columns)
            self.assertIn("p_adj", pairwise_df.columns)
            self.assertTrue(pairwise_df["delta_ci_lo_raw"].isna().all())
            self.assertTrue(pairwise_df["delta_ci_hi_raw"].isna().all())
            self.assertEqual(set(pairwise_df["handle0"]), {"baseline"})
            self.assertEqual(set(pairwise_df["handle1"]), {"candidate"})

    def test_one_sided_permutation_respects_lower_is_better_metrics(self) -> None:
        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.5, 1.5, 2.5, 3.5])
        y1 = np.array([0.0, 1.0, 2.0, 3.0])

        p_rmse = _paired_permutation_pval(
            y_true,
            y0,
            y1,
            metric_fn=_rmse,
            higher_is_better=False,
            n_samples=50,
            seed=123,
            alternative="one-sided",
            require_two_classes=False,
        )
        p_neg_rmse = _paired_permutation_pval(
            y_true,
            y0,
            y1,
            metric_fn=lambda yt, yp: -_rmse(yt, yp),
            higher_is_better=True,
            n_samples=50,
            seed=123,
            alternative="one-sided",
            require_two_classes=False,
        )

        self.assertAlmostEqual(p_rmse, p_neg_rmse, places=12)

    def test_derive_pred_paths_supports_regression_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_loc = tmp / "model-checkpoint"
            model_loc.mkdir()
            test_dir = tmp / "demo-tokenized" / "test"
            test_dir.mkdir(parents=True)
            pred_path = test_dir / "reg_ridge_regression-preds-model-checkpoint.pkl"
            pred_path.write_bytes(b"payload")

            paths = _derive_pred_paths(
                data_dir=tmp,
                data_versions=["demo"],
                classifier="ridge_regression",
                task_type="regression",
                model_loc=model_loc,
            )

            self.assertEqual(paths, [pred_path])


if __name__ == "__main__":
    unittest.main()
