"""
Automated Unit Tests for Phase 6: Statistical Validation.
Verifies all Phase 6 deliverables, statistical results, hypothesis test outcomes,
confidence intervals, and sensitivity analysis.
"""

import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import config


class TestPhase6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phase6_dir = config.PHASE6_OUTPUT_DIR
        cls.paired_csv = cls.phase6_dir / "paired_test_summary.csv"
        cls.boot_csv = cls.phase6_dir / "bootstrap_ci_summary.csv"
        cls.report_md = cls.phase6_dir / "significance_report.md"
        cls.manifest_json = cls.phase6_dir / "manifest.json"
        cls.script_path = cls.phase6_dir / "statistical_validation.py"

        cls.df_paired = pd.read_csv(cls.paired_csv)
        cls.df_boot = pd.read_csv(cls.boot_csv)
        with open(cls.manifest_json, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

    def test_artifacts_existence(self):
        """Verify all mandated Phase 6 output files exist and are non-empty."""
        self.assertTrue(self.paired_csv.exists())
        self.assertTrue(self.boot_csv.exists())
        self.assertTrue(self.report_md.exists())
        self.assertTrue(self.manifest_json.exists())
        self.assertTrue(self.script_path.exists())

        self.assertGreater(self.paired_csv.stat().st_size, 1000)
        self.assertGreater(self.boot_csv.stat().st_size, 1000)
        self.assertGreater(self.report_md.stat().st_size, 5000)
        self.assertGreater(self.manifest_json.stat().st_size, 1500)

    def test_primary_comparisons_statistics(self):
        """Verify primary comparisons (M5 vs M6 and M5 vs M3) match audit targets."""
        row_m6 = self.df_paired[self.df_paired["comparison"] == "M5_vs_M6"].iloc[0]
        row_m3 = self.df_paired[self.df_paired["comparison"] == "M5_vs_M3"].iloc[0]

        # Mean differences
        self.assertAlmostEqual(row_m6["mean_difference"], 0.019258, places=5)
        self.assertAlmostEqual(row_m3["mean_difference"], 0.014758, places=5)

        # Wilcoxon raw p-values
        self.assertAlmostEqual(row_m6["wilcoxon_raw_p"], 0.5543, places=3)
        self.assertAlmostEqual(row_m3["wilcoxon_raw_p"], 0.3259, places=3)

        # Holm-adjusted p-values (both non-significant at alpha=0.05)
        self.assertAlmostEqual(row_m6["wilcoxon_holm_adjusted_p"], 0.6519, places=3)
        self.assertAlmostEqual(row_m3["wilcoxon_holm_adjusted_p"], 0.6519, places=3)

        self.assertFalse(row_m6["significant_wilcoxon_alpha_005"])
        self.assertFalse(row_m3["significant_wilcoxon_alpha_005"])

    def test_query_wins_losses_counts(self):
        """Verify positive, tie, and negative query counts."""
        row_m6 = self.df_paired[self.df_paired["comparison"] == "M5_vs_M6"].iloc[0]
        row_m3 = self.df_paired[self.df_paired["comparison"] == "M5_vs_M3"].iloc[0]

        self.assertEqual(row_m6["positive_count_wins"], 53)
        self.assertEqual(row_m6["zero_count_ties"], 0)
        self.assertEqual(row_m6["negative_count_losses"], 47)

        self.assertEqual(row_m3["positive_count_wins"], 51)
        self.assertEqual(row_m3["zero_count_ties"], 6)
        self.assertEqual(row_m3["negative_count_losses"], 43)

    def test_bootstrap_confidence_intervals(self):
        """Verify category-stratified bootstrap 95% confidence intervals."""
        boot_m6 = self.df_boot[self.df_boot["comparison"] == "M5_vs_M6"].iloc[0]
        boot_m3 = self.df_boot[self.df_boot["comparison"] == "M5_vs_M3"].iloc[0]

        self.assertGreater(boot_m6["ci_95_percentile_lower"], 0.010)
        self.assertLess(boot_m6["ci_95_percentile_upper"], 0.030)

        self.assertGreater(boot_m3["ci_95_percentile_lower"], 0.005)
        self.assertLess(boot_m3["ci_95_percentile_upper"], 0.025)

    def test_flowers_driver_effect_inversion(self):
        """Verify that without Flowers, mean differences are negative and CIs cross zero."""
        row_m6_nf = self.df_paired[self.df_paired["comparison"] == "M5_vs_M6_no_flowers"].iloc[0]
        row_m3_nf = self.df_paired[self.df_paired["comparison"] == "M5_vs_M3_no_flowers"].iloc[0]

        self.assertLess(row_m6_nf["mean_difference"], 0.0)
        self.assertLess(row_m3_nf["mean_difference"], 0.0)

        boot_m6_nf = self.df_boot[self.df_boot["comparison"] == "M5_vs_M6_no_flowers"].iloc[0]
        boot_m3_nf = self.df_boot[self.df_boot["comparison"] == "M5_vs_M3_no_flowers"].iloc[0]

        self.assertLess(boot_m6_nf["ci_95_percentile_lower"], 0.0)
        self.assertGreater(boot_m6_nf["ci_95_percentile_upper"], 0.0)

        self.assertLess(boot_m3_nf["ci_95_percentile_lower"], 0.0)
        self.assertGreater(boot_m3_nf["ci_95_percentile_upper"], 0.0)

    def test_manifest_metadata_and_integrity(self):
        """Verify manifest payload contains all required tracking metadata."""
        self.assertEqual(self.manifest["phase"], 6)
        self.assertIn("input_artifacts", self.manifest)
        self.assertIn("execution_parameters", self.manifest)
        self.assertIn("formal_statistical_conclusion", self.manifest)
        self.assertIn("limitations", self.manifest)

        self.assertFalse(self.manifest["formal_statistical_conclusion"]["is_m5_statistically_superior"])
        self.assertGreaterEqual(len(self.manifest["limitations"]), 3)


if __name__ == "__main__":
    unittest.main()
