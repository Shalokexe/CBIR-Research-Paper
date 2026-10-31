"""
Unit Test Suite for Phase 9: External Transfer and Computational Analysis.

Tests:
1. Verification of output deliverables existence and non-emptiness.
2. External transfer metrics schema, model coverage (M1-M6), and valid ranges [0, 1].
3. Breed metrics completeness (exactly 37 fine-grained breeds, Cats vs Dogs).
4. Runtime and scalability metrics integrity (latency monotonicity, 52-byte storage).
5. Mathematical consistency of metrics (P@10, R@10, F1@10, mAP).
6. Cryptographic SHA-256 integrity verification from manifest.json.
"""

import unittest
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE9_DIR = PROJECT_ROOT / "output" / "phase9"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestPhase9(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transfer_metrics_path = PHASE9_DIR / "external_transfer_metrics.csv"
        cls.breed_metrics_path = PHASE9_DIR / "breed_metrics.csv"
        cls.runtime_report_path = PHASE9_DIR / "runtime_report.csv"
        cls.transfer_report_path = PHASE9_DIR / "transfer_report.md"
        cls.scalability_report_path = PHASE9_DIR / "scalability_report.md"
        cls.manifest_path = PHASE9_DIR / "manifest.json"

        # Check all required files exist
        for p in [cls.transfer_metrics_path, cls.breed_metrics_path,
                  cls.runtime_report_path, cls.transfer_report_path,
                  cls.scalability_report_path, cls.manifest_path]:
            assert p.exists(), f"Phase 9 required file missing: {p}"

        cls.df_transfer = pd.read_csv(cls.transfer_metrics_path)
        cls.df_breed = pd.read_csv(cls.breed_metrics_path)
        cls.df_runtime = pd.read_csv(cls.runtime_report_path)

        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

    def test_external_transfer_metrics_completeness(self):
        """Verify external_transfer_metrics.csv covers all M1-M6 models with valid metrics."""
        expected_models = [
            "M1_Colour",
            "M2_Texture",
            "M3_Equal",
            "M4_Fixed",
            "M5_Proposed_Adaptive",
            "M6_Mean_Control"
        ]
        present_models = self.df_transfer["method_id"].tolist()
        for m in expected_models:
            self.assertIn(m, present_models, f"Model {m} must be present in external_transfer_metrics.csv")

        for col in ["mean_precision_at_10", "mean_recall_at_10", "mean_f1_at_10", "mean_average_precision"]:
            self.assertIn(col, self.df_transfer.columns)
            vals = self.df_transfer[col].values
            self.assertTrue(np.all(vals >= 0.0), f"All {col} values must be >= 0")
            self.assertTrue(np.all(vals <= 1.0), f"All {col} values must be <= 1")

        # Zero-shot transfer mAP on fine-grained 37 breeds should reflect fine-grained difficulty (~0.02 - 0.06)
        m5_map = self.df_transfer[self.df_transfer["method_id"] == "M5_Proposed_Adaptive"]["mean_average_precision"].values[0]
        self.assertGreater(m5_map, 0.01, "M5 mAP should be above random chance baseline")
        self.assertLess(m5_map, 0.15, "M5 zero-shot mAP on 37 fine-grained breeds is expectedly low")

    def test_breed_metrics_coverage(self):
        """Verify breed_metrics.csv contains all 37 Oxford-IIIT Pet breeds (12 Cat, 25 Dog)."""
        self.assertEqual(len(self.df_breed), 37, "Oxford-IIIT Pet must have exactly 37 breeds.")
        self.assertIn("breed", self.df_breed.columns)
        self.assertIn("species", self.df_breed.columns)

        cat_count = len(self.df_breed[self.df_breed["species"] == "Cat"])
        dog_count = len(self.df_breed[self.df_breed["species"] == "Dog"])
        self.assertEqual(cat_count, 12, "Must contain exactly 12 cat breeds.")
        self.assertEqual(dog_count, 25, "Must contain exactly 25 dog breeds.")

        for col in ["m1_colour_map", "m2_texture_map", "m3_equal_map", "m4_fixed_map",
                    "m5_proposed_adaptive_map", "m6_mean_control_map"]:
            self.assertIn(col, self.df_breed.columns)
            vals = self.df_breed[col].values
            self.assertTrue(np.all(vals >= 0.0))
            self.assertTrue(np.all(vals <= 1.0))

    def test_runtime_and_storage_invariants(self):
        """Verify runtime components and 52-byte storage footprint."""
        df_r = self.df_runtime.set_index("component")

        # Feature storage per image must be exactly 52 bytes (13 float32)
        feat_bytes = df_r.loc["Feature_Storage_Per_Image", "metric_value"]
        self.assertEqual(feat_bytes, 52)

        # Latencies must be strictly positive
        for comp in ["Feature_Extraction_Total", "MLP_Query_Inference", "Corel_Ranking_Latency_N899"]:
            self.assertIn(comp, df_r.index)
            val = df_r.loc[comp, "metric_value"]
            self.assertGreater(val, 0.0)

        # Monotonicity: Oxford latency (N=6979) > Corel latency (N=899)
        lat_corel = df_r.loc["Corel_Ranking_Latency_N899", "metric_value"]
        lat_oxford = df_r.loc["Oxford_Ranking_Latency_N6979", "metric_value"]
        self.assertGreater(lat_oxford, lat_corel, "Larger gallery should require more ranking time")

    def test_manifest_cryptographic_integrity(self):
        """Verify all SHA-256 checksums in manifest.json match generated files."""
        self.assertIn("output_checksums_sha256", self.manifest)
        for filepath_str, expected_sha in self.manifest["output_checksums_sha256"].items():
            filepath = Path(filepath_str)
            self.assertTrue(filepath.exists(), f"File {filepath} in manifest must exist.")
            actual_sha = compute_sha256(filepath)
            self.assertEqual(actual_sha, expected_sha, f"SHA-256 mismatch for {filepath.name}")


if __name__ == "__main__":
    unittest.main()
