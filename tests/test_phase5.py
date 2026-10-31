"""
Automated Unit Tests for Phase 5: Corrected Corel-1K Benchmark Evaluation.
Verifies artifact existence, row counts, reference parity, weight simplex constraints,
and statistical audit results.
"""

import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import config


class TestPhase5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phase5_dir = config.PHASE5_OUTPUT_DIR
        cls.results_json_path = cls.phase5_dir / "results.json"
        cls.per_query_path = cls.phase5_dir / "per_query_metrics.csv"
        cls.cat_metrics_path = cls.phase5_dir / "category_metrics.csv"
        cls.query_weights_path = cls.phase5_dir / "query_weights.csv"
        cls.rankings_path = cls.phase5_dir / "rankings.npz"
        cls.report_path = cls.phase5_dir / "evaluation_report.md"

        with open(cls.results_json_path, "r", encoding="utf-8") as f:
            cls.results = json.load(f)

        cls.df_queries = pd.read_csv(cls.per_query_path)
        cls.df_cat = pd.read_csv(cls.cat_metrics_path)
        cls.df_weights = pd.read_csv(cls.query_weights_path)

    def test_artifact_existence(self):
        """Verify all mandated Phase 5 output files exist and are non-empty."""
        self.assertTrue(self.results_json_path.exists())
        self.assertTrue(self.per_query_path.exists())
        self.assertTrue(self.cat_metrics_path.exists())
        self.assertTrue(self.query_weights_path.exists())
        self.assertTrue(self.rankings_path.exists())
        self.assertTrue(self.report_path.exists())

        self.assertGreater(self.results_json_path.stat().st_size, 1000)
        self.assertGreater(self.per_query_path.stat().st_size, 50000)
        self.assertGreater(self.cat_metrics_path.stat().st_size, 3000)
        self.assertGreater(self.query_weights_path.stat().st_size, 8000)
        self.assertGreater(self.rankings_path.stat().st_size, 2000000)
        self.assertGreater(self.report_path.stat().st_size, 5000)

    def test_row_counts_and_structure(self):
        """Verify dimensions: 600 query rows, 60 category rows, 100 weight rows."""
        self.assertEqual(len(self.df_queries), 600)
        self.assertEqual(len(self.df_cat), 60)
        self.assertEqual(len(self.df_weights), 100)

        # 6 models
        self.assertEqual(set(self.df_queries["model"].unique()), {"M1", "M2", "M3", "M4", "M5", "M6"})
        self.assertEqual(set(self.df_cat["model"].unique()), {"M1", "M2", "M3", "M4", "M5", "M6"})

        # 10 categories
        self.assertEqual(len(self.df_cat["category"].unique()), 10)

    def test_reference_benchmark_parity(self):
        """Verify exact mAP parity against published benchmark values."""
        perf = self.results["aggregate_model_performance"]

        self.assertAlmostEqual(perf["M1"]["mAP"], 0.404462, places=6)
        self.assertAlmostEqual(perf["M2"]["mAP"], 0.350816, places=6)
        self.assertAlmostEqual(perf["M3"]["mAP"], 0.465036, places=6)
        self.assertAlmostEqual(perf["M4"]["mAP"], 0.433659, places=6)
        self.assertAlmostEqual(perf["M5"]["mAP"], 0.479794, places=6)
        self.assertAlmostEqual(perf["M6"]["mAP"], 0.460536, places=6)

        # Precision@10 reference values
        self.assertAlmostEqual(perf["M3"]["precision_at_10"], 0.6990, places=4)
        self.assertAlmostEqual(perf["M5"]["precision_at_10"], 0.7100, places=4)
        self.assertAlmostEqual(perf["M6"]["precision_at_10"], 0.6960, places=4)

    def test_query_weights_simplex_constraint(self):
        """Verify non-negativity and sum-to-one constraint on predicted query weights."""
        wc = self.df_weights["m5_color_weight"].to_numpy()
        wt = self.df_weights["m5_texture_weight"].to_numpy()

        self.assertTrue(np.all(wc >= 0.0))
        self.assertTrue(np.all(wt >= 0.0))
        np.testing.assert_allclose(wc + wt, 1.0, atol=1e-6)

        # Variance must be non-zero
        self.assertGreater(float(np.std(wc)), 0.1)

    def test_statistical_test_audit_values(self):
        """Verify Wilcoxon signed-rank and Holm-adjusted values match audit standards."""
        tests = self.results["statistical_hypothesis_tests"]

        # Mean paired differences
        self.assertAlmostEqual(tests["M5_vs_M3"]["mean_paired_difference"], 0.014758, places=5)
        self.assertAlmostEqual(tests["M5_vs_M6"]["mean_paired_difference"], 0.019258, places=5)

        # Non-significance confirmed
        self.assertFalse(tests["M5_vs_M3"]["statistically_significant_alpha_005"])
        self.assertFalse(tests["M5_vs_M6"]["statistically_significant_alpha_005"])

        self.assertAlmostEqual(tests["M5_vs_M3"]["wilcoxon_p_twosided"], 0.3259, places=3)
        self.assertAlmostEqual(tests["M5_vs_M6"]["wilcoxon_p_twosided"], 0.5543, places=3)
        self.assertAlmostEqual(tests["M5_vs_M3"]["holm_bonferroni_adjusted_p"], 0.6519, places=3)
        self.assertAlmostEqual(tests["M5_vs_M6"]["holm_bonferroni_adjusted_p"], 0.6519, places=3)

    def test_flowers_driver_effect(self):
        """Verify that excluding Flowers flips the aggregate difference to negative."""
        sens = self.results["category_sensitivity_analysis"]["flowers_driver_effect"]

        # Flowers-only difference is heavily positive
        self.assertGreater(sens["flowers_mean_diff_m5_m3"], 0.15)
        self.assertGreater(sens["flowers_mean_diff_m5_m6"], 0.15)

        # Without Flowers difference is negative
        self.assertLess(sens["mean_diff_without_flowers_m5_m3"], 0.0)
        self.assertLess(sens["mean_diff_without_flowers_m5_m6"], 0.0)


if __name__ == "__main__":
    unittest.main()
