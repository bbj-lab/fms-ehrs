from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from fms_ehrs.scripts.aggregate_version_preds import main as aggregate_main


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
            self.assertIn("p_raw", pairwise_df.columns)
            self.assertIn("p_adj", pairwise_df.columns)
            self.assertEqual(set(pairwise_df["handle0"]), {"baseline"})
            self.assertEqual(set(pairwise_df["handle1"]), {"candidate"})


if __name__ == "__main__":
    unittest.main()
