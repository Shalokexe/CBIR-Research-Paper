"""
Unit Test Suite for Phase 8: Handcrafted and Deep-Learning Baselines.

Tests:
1. Verification of output deliverables existence and non-emptiness.
2. Baseline metrics schema, completeness, and mathematical validity (P@10, R@10, F1@10, mAP in [0, 1]).
3. Deep-learning inventory completeness (ResNet-50, EfficientNet-B0, CLIP ViT-B/32).
4. Paired statistical comparison integrity (Wilcoxon tests, Holm-Bonferroni correction monotonic adjustments).
5. Storage and latency efficiency invariants (M5 storage << D2/D3/D5).
6. Per-query and per-category completeness (100 queries, 10 categories across all 8 evaluated methods).
7. Cryptographic integrity and SHA-256 manifest verification.
"""

import unittest
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE8_DIR = PROJECT_ROOT / "output" / "phase8"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestPhase8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_metrics_path = PHASE8_DIR / "baseline_metrics.csv"
        cls.per_query_metrics_path = PHASE8_DIR / "per_query_metrics.csv"
        cls.category_metrics_path = PHASE8_DIR / "category_metrics.csv"
        cls.paired_comparisons_path = PHASE8_DIR / "paired_comparisons.csv"
        cls.runtime_comparison_path = PHASE8_DIR / "runtime_comparison.csv"
        cls.feature_inventory_path = PHASE8_DIR / "feature_inventory.json"
        cls.report_path = PHASE8_DIR / "phase_d_report.md"
        cls.manifest_path = PHASE8_DIR / "manifest.json"

        # Check all core files exist
        for p in [cls.baseline_metrics_path, cls.per_query_metrics_path,
                  cls.category_metrics_path, cls.paired_comparisons_path,
                  cls.runtime_comparison_path, cls.feature_inventory_path,
                  cls.report_path, cls.manifest_path]:
            assert p.exists(), f"Phase 8 required file missing: {p}"

        cls.df_baseline = pd.read_csv(cls.baseline_metrics_path)
        cls.df_per_query = pd.read_csv(cls.per_query_metrics_path)
        cls.df_category = pd.read_csv(cls.category_metrics_path)
        cls.df_paired = pd.read_csv(cls.paired_comparisons_path)
        cls.df_runtime = pd.read_csv(cls.runtime_comparison_path)

        with open(cls.feature_inventory_path, "r", encoding="utf-8") as f:
            cls.inventory = json.load(f)

    def test_baseline_metrics_completeness(self):
        """Verify baseline_metrics.csv contains all required baselines and valid metrics."""
        expected_methods = [
            "M5_proposed",
            "D1_LBP",
            "D2_ResNet50",
            "D3_EfficientNet",
            "D4_Fixed_M3",
            "D4_Fixed_M4",
            "D4_Fixed_M6",
            "D5_CLIP"
        ]
        present_methods = self.df_baseline["method_id"].tolist()
        for m in expected_methods:
            self.assertIn(m, present_methods, f"Method {m} must be present in baseline_metrics.csv")

        # Metric ranges
        for col in ["mean_precision_at_10", "mean_recall_at_10", "mean_f1_at_10", "mean_average_precision"]:
            self.assertIn(col, self.df_baseline.columns)
            vals = self.df_baseline[col].values
            self.assertTrue(np.all(vals >= 0.0), f"All {col} values must be >= 0")
            self.assertTrue(np.all(vals <= 1.0), f"All {col} values must be <= 1")

    def test_per_query_metrics_coverage(self):
        """Verify per-query metrics has 100 queries for all 8 methods (800 rows)."""
        self.assertEqual(len(self.df_per_query), 800)
        methods = self.df_per_query["method_id"].unique()
        self.assertEqual(len(methods), 8)
        for m in methods:
            sub = self.df_per_query[self.df_per_query["method_id"] == m]
            self.assertEqual(len(sub), 100, f"Method {m} must have exactly 100 per-query records.")

    def test_category_metrics_coverage(self):
        """Verify category metrics has 10 categories for all 8 methods (80 rows)."""
        self.assertEqual(len(self.df_category), 80)
        methods = self.df_category["method_id"].unique()
        self.assertEqual(len(methods), 8)
        categories = self.df_category["category"].unique()
        self.assertEqual(len(categories), 10)

    def test_deep_learning_inventory_schema(self):
        """Verify exact model variant, layer, resolution, norm, dim, and distance for DL models."""
        dl_keys = ["D2_ResNet50", "D3_EfficientNet", "D5_CLIP"]
        self.assertIn("methods", self.inventory)
        methods_inv = self.inventory["methods"]

        for key in dl_keys:
            self.assertIn(key, methods_inv, f"{key} must be in feature inventory methods")
            info = methods_inv[key]
            self.assertIn("model_variant", info)
            self.assertIn("checkpoint_source", info)
            self.assertIn("feature_extraction_layer", info)
            self.assertIn("input_resolution", info)
            self.assertIn("normalization", info)
            self.assertIn("feature_dimensionality", info)
            self.assertIn("distance_metric", info)
            self.assertIn("training_status", info)
            self.assertIn("frozen", info["training_status"].lower())

    def test_paired_statistical_comparisons(self):
        """Verify statistical testing against M5 Proposed CBIR with Holm correction."""
        self.assertGreaterEqual(len(self.df_paired), 7)
        for col in ["comparison", "baseline_id", "mean_paired_diff",
                    "wilcoxon_stat_w", "raw_p_value", "holm_adjusted_p_value"]:
            self.assertIn(col, self.df_paired.columns)

        # Holm adjusted p-values must be in [0, 1]
        holm_p = self.df_paired["holm_adjusted_p_value"].values
        self.assertTrue(np.all((holm_p >= 0.0) & (holm_p <= 1.0)))

        # Holm adjustment monotonicity: holm_p >= raw_p
        raw_p = self.df_paired["raw_p_value"].values
        self.assertTrue(np.all(holm_p >= raw_p - 1e-9))

    def test_runtime_and_storage_tradeoffs(self):
        """Verify runtime and storage invariants showing M5 niche vs DL."""
        df_r = self.df_runtime.set_index("method_id")

        # Storage invariant: M5 is 52 bytes (13 float32)
        m5_bytes = df_r.loc["M5_proposed", "feature_bytes_per_image"]
        self.assertEqual(m5_bytes, 52)

        resnet_bytes = df_r.loc["D2_ResNet50", "feature_bytes_per_image"]
        self.assertEqual(resnet_bytes, 8192)  # 2048 float32

        clip_bytes = df_r.loc["D5_CLIP", "feature_bytes_per_image"]
        self.assertEqual(clip_bytes, 2048)  # 512 float32

        self.assertLess(m5_bytes, resnet_bytes)
        self.assertLess(m5_bytes, clip_bytes)

        # Latency invariant: M5 feature extraction is significantly faster than ResNet-50 and CLIP
        m5_extract = df_r.loc["M5_proposed", "extraction_latency_ms_per_image"]
        resnet_extract = df_r.loc["D2_ResNet50", "extraction_latency_ms_per_image"]
        clip_extract = df_r.loc["D5_CLIP", "extraction_latency_ms_per_image"]

        self.assertLess(m5_extract, resnet_extract)
        self.assertLess(m5_extract, clip_extract)

    def test_manifest_cryptographic_integrity(self):
        """Verify manifest.json contains valid SHA-256 checksums matching file contents."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("output_checksums_sha256", manifest)
        self.assertIn("input_checksums_sha256", manifest)

        for filepath_str, expected_sha in manifest["output_checksums_sha256"].items():
            filepath = Path(filepath_str)
            self.assertTrue(filepath.exists(), f"File {filepath} in manifest must exist.")
            actual_sha = compute_sha256(filepath)
            self.assertEqual(actual_sha, expected_sha, f"SHA-256 mismatch for {filepath.name}")


if __name__ == "__main__":
    unittest.main()
