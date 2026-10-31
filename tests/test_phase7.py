"""
Unit Test Suite for Phase 7: Gallery-Composition Sensitivity Analysis.

Tests:
1. Scenario definitions schema and completeness (16 scenarios).
2. C2 Nested stratified gallery subset hierarchy (25% ⊂ 50% ⊂ 75% ⊂ 100%).
3. C1 Duplicate-group query exclusion invariants (99 queries, 899 gallery).
4. Metric range invariants and mathematical consistency (P@10, R@10, F1@10, mAP).
5. C3 Flowers concentration and LOCO robustness verification.
6. Execution manifest cryptographic integrity and checksum verification.
"""

import unittest
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE7_DIR = PROJECT_ROOT / "output" / "phase7"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestPhase7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defs_path = PHASE7_DIR / "scenario_definitions.json"
        cls.scenario_metrics_path = PHASE7_DIR / "scenario_metrics.csv"
        cls.per_query_metrics_path = PHASE7_DIR / "per_query_scenario_metrics.csv"
        cls.category_metrics_path = PHASE7_DIR / "category_scenario_metrics.csv"
        cls.report_path = PHASE7_DIR / "phase_c_report.md"
        cls.manifest_path = PHASE7_DIR / "manifest.json"

        # Check all core files exist
        for p in [cls.defs_path, cls.scenario_metrics_path, cls.per_query_metrics_path,
                  cls.category_metrics_path, cls.report_path, cls.manifest_path]:
            assert p.exists(), f"Phase 7 required file missing: {p}"

        with open(cls.defs_path, "r", encoding="utf-8") as f:
            cls.scenarios = json.load(f)

        cls.df_scenarios = pd.read_csv(cls.scenario_metrics_path)
        cls.df_per_query = pd.read_csv(cls.per_query_metrics_path)
        cls.df_category = pd.read_csv(cls.category_metrics_path)

    def test_scenario_definitions_structure(self):
        """Verify scenario definitions JSON schema and scenario counts."""
        self.assertGreaterEqual(len(self.scenarios), 16, "Must define at least 16 sensitivity scenarios.")

        expected_scenarios = [
            "baseline_corrected",
            "c1_duplicate_purged_query",
            "c2_gallery_75pct",
            "c2_gallery_50pct",
            "c2_gallery_25pct",
            "c3_without_flowers_query",
            "c3_without_flowers_full"
        ]
        for s in expected_scenarios:
            self.assertIn(s, self.scenarios, f"Scenario {s} must be present in definitions.")

        # Check required fields
        for sid, sdata in self.scenarios.items():
            self.assertIn("scenario_id", sdata)
            self.assertIn("name", sdata)
            self.assertIn("scenario_group", sdata)
            self.assertIn("gallery_size", sdata)
            self.assertIn("query_count", sdata)
            self.assertIn("per_category_gallery_counts", sdata)
            self.assertIn("per_category_query_counts", sdata)
            self.assertIn("is_nested", sdata)

    def test_c2_nested_stratified_invariants(self):
        """Verify nested stratified gallery hierarchy: 25% ⊂ 50% ⊂ 75% ⊂ 100%."""
        s25 = self.scenarios["c2_gallery_25pct"]
        s50 = self.scenarios["c2_gallery_50pct"]
        s75 = self.scenarios["c2_gallery_75pct"]
        s100 = self.scenarios["baseline_corrected"]

        self.assertEqual(s25["gallery_size"], 220)
        self.assertEqual(s50["gallery_size"], 449)
        self.assertEqual(s75["gallery_size"], 679)
        self.assertEqual(s100["gallery_size"], 899)

        # Check query counts remain fixed at 100
        for s in [s25, s50, s75, s100]:
            self.assertEqual(s["query_count"], 100)

        # Check per-category monotonicity
        for cat in s100["per_category_gallery_counts"]:
            cnt25 = s25["per_category_gallery_counts"][cat]
            cnt50 = s50["per_category_gallery_counts"][cat]
            cnt75 = s75["per_category_gallery_counts"][cat]
            cnt100 = s100["per_category_gallery_counts"][cat]

            self.assertLessEqual(cnt25, cnt50)
            self.assertLessEqual(cnt50, cnt75)
            self.assertLessEqual(cnt75, cnt100)

    def test_c1_duplicate_group_purged_scenario(self):
        """Verify duplicate group query removal invariants."""
        s_c1 = self.scenarios["c1_duplicate_purged_query"]
        self.assertEqual(s_c1["query_count"], 99)
        self.assertEqual(s_c1["gallery_size"], 899)
        self.assertIn("training_set/horses/776.jpg", s_c1["excluded_queries"])
        self.assertIn("training_set/horses/761.jpg", s_c1["excluded_gallery_paths"])

        # Check that query 776.jpg is not evaluated in per_query_scenario_metrics
        c1_queries = self.df_per_query[self.df_per_query["scenario_id"] == "c1_duplicate_purged_query"]
        self.assertEqual(len(c1_queries), 99 * 6)
        self.assertNotIn("training_set/horses/776.jpg", c1_queries["relative_path"].values)

    def test_metric_invariants_and_ranges(self):
        """Verify metric bounds, unrounded calculations, and gallery size recall inflation."""
        # Check range bounds in [0, 1]
        for col in ["mean_precision_at_10", "mean_recall_at_10", "mean_f1_at_10", "query_level_map", "macro_category_map"]:
            vals = self.df_scenarios[col].values
            self.assertTrue(np.all(vals >= 0.0), f"{col} has negative values")
            self.assertTrue(np.all(vals <= 1.0), f"{col} exceeds 1.0")

        # In baseline, macro_category_map must match query_level_map (balanced queries)
        base_df = self.df_scenarios[self.df_scenarios["scenario_id"] == "baseline_corrected"]
        for _, row in base_df.iterrows():
            self.assertAlmostEqual(
                row["query_level_map"],
                row["macro_category_map"],
                places=6,
                msg=f"Macro AP and query mAP mismatch for {row['model']}"
            )

        # Mathematical property: Recall@10 in 25% gallery must be strictly greater than Recall@10 in 100% gallery
        for m in ["M1", "M2", "M3", "M4", "M5", "M6"]:
            r25 = self.df_scenarios[(self.df_scenarios["scenario_id"] == "c2_gallery_25pct") & (self.df_scenarios["model"] == m)]["mean_recall_at_10"].values[0]
            r100 = self.df_scenarios[(self.df_scenarios["scenario_id"] == "baseline_corrected") & (self.df_scenarios["model"] == m)]["mean_recall_at_10"].values[0]
            self.assertGreater(r25, r100 * 2.0, f"Recall@10 at 25% gallery should be > 2x recall at 100% for {m}")

    def test_c3_flowers_driver_concentration(self):
        """Verify empirical concentration of M5 advantage in Flowers category."""
        # In baseline, M5 beats M6 and M3
        base_df = self.df_scenarios[self.df_scenarios["scenario_id"] == "baseline_corrected"]
        m5_base = base_df[base_df["model"] == "M5"]["query_level_map"].values[0]
        m6_base = base_df[base_df["model"] == "M6"]["query_level_map"].values[0]
        m3_base = base_df[base_df["model"] == "M3"]["query_level_map"].values[0]

        self.assertGreater(m5_base, m6_base)
        self.assertGreater(m5_base, m3_base)

        # In without_flowers_query, M5 advantage turns negative
        no_flow_df = self.df_scenarios[self.df_scenarios["scenario_id"] == "c3_without_flowers_query"]
        m5_nf = no_flow_df[no_flow_df["model"] == "M5"]["query_level_map"].values[0]
        m6_nf = no_flow_df[no_flow_df["model"] == "M6"]["query_level_map"].values[0]
        m3_nf = no_flow_df[no_flow_df["model"] == "M3"]["query_level_map"].values[0]

        self.assertLess(m5_nf, m6_nf, "Without Flowers, M5 must underperform M6")
        self.assertLess(m5_nf, m3_nf, "Without Flowers, M5 must underperform M3")

        # In LOCO scenarios for other 9 categories, M5 retains positive advantage
        loco_categories = [
            "beaches", "bus", "dinosaurs", "elephants", "foods",
            "horses", "monuments", "mountains_and_snow", "peolpe_and_villages_in_Africa"
        ]
        for c in loco_categories:
            sid = f"c3_loco_without_{c}"
            loco_df = self.df_scenarios[self.df_scenarios["scenario_id"] == sid]
            m5_loco = loco_df[loco_df["model"] == "M5"]["query_level_map"].values[0]
            m6_loco = loco_df[loco_df["model"] == "M6"]["query_level_map"].values[0]
            self.assertGreater(
                m5_loco, m6_loco,
                f"LOCO without {c} should preserve positive M5 advantage over M6"
            )

    def test_manifest_integrity_and_checksums(self):
        """Verify manifest.json contains valid metadata and matching SHA-256 checksums."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["phase"], "Phase 7 — Gallery-Composition Sensitivity")
        self.assertIn("execution_duration_seconds", manifest)
        self.assertIn("environment", manifest)
        self.assertIn("output_checksums_sha256", manifest)

        # Verify output file hashes match
        for filepath_str, recorded_hash in manifest["output_checksums_sha256"].items():
            fpath = Path(filepath_str)
            self.assertTrue(fpath.exists(), f"Output file listed in manifest does not exist: {fpath}")
            current_hash = compute_sha256(fpath)
            self.assertEqual(
                current_hash, recorded_hash,
                f"SHA-256 mismatch for {fpath.name}: {current_hash} vs recorded {recorded_hash}"
            )


if __name__ == "__main__":
    unittest.main()
