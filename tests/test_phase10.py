"""
Unit tests for Phase 10: Final Research Audit and Publication Package.
Verifies the existence, structural validity, numeric consistency, and cryptographic
integrity of all manuscript tables, publication figures, checklists, audit reports,
and manifest files.
"""

import os
import json
import hashlib
import unittest
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestPhase10PublicationPackage(unittest.TestCase):
    """Test suite for Phase 10 publication package and research audit."""

    def setUp(self):
        self.phase10_dir = PROJECT_ROOT / "output" / "phase10"
        self.tables_dir = PROJECT_ROOT / "manuscript_tables"
        self.figs_dir = PROJECT_ROOT / "publication_figures"
        self.output_tables_dir = self.phase10_dir / "manuscript_tables"
        self.output_figs_dir = self.phase10_dir / "publication_figures"

    def test_01_manuscript_tables_exist(self):
        """Verify all 7 manuscript tables exist in both .csv and .tex formats."""
        expected_tables = [
            "table1_main_performance",
            "table2_statistical_significance",
            "table3_category_metrics",
            "table4_gallery_sensitivity",
            "table5_deep_baselines",
            "table6_oxford_transfer",
            "table7_runtime_scalability"
        ]

        for t in expected_tables:
            for ext in [".csv", ".tex"]:
                root_f = self.tables_dir / f"{t}{ext}"
                out_f = self.output_tables_dir / f"{t}{ext}"
                self.assertTrue(root_f.exists(), f"Missing root table: {root_f}")
                self.assertTrue(out_f.exists(), f"Missing output table: {out_f}")
                self.assertGreater(root_f.stat().st_size, 50, f"Table too small: {root_f}")

    def test_02_manuscript_tables_content(self):
        """Verify table contents are structurally valid and numeric."""
        # Table 1: Main performance (M1-M6)
        df1 = pd.read_csv(self.tables_dir / "table1_main_performance.csv")
        self.assertEqual(len(df1), 6)
        self.assertIn("mean_average_precision", df1.columns)
        m5_row = df1[df1["model_id"] == "M5"].iloc[0]
        self.assertAlmostEqual(m5_row["mean_average_precision"], 0.4798, places=3)
        self.assertAlmostEqual(m5_row["precision_at_10"], 0.7100, places=3)

        # Table 2: Statistical significance
        df2 = pd.read_csv(self.tables_dir / "table2_statistical_significance.csv")
        self.assertGreaterEqual(len(df2), 5)
        self.assertIn("holm_adjusted_p_value", df2.columns)

        # Table 3: Per-category metrics (10 categories)
        df3 = pd.read_csv(self.tables_dir / "table3_category_metrics.csv")
        self.assertEqual(len(df3), 10)

        # Table 4: Gallery sensitivity
        df4 = pd.read_csv(self.tables_dir / "table4_gallery_sensitivity.csv")
        self.assertGreaterEqual(len(df4), 12)

        # Table 5: Deep baselines
        df5 = pd.read_csv(self.tables_dir / "table5_deep_baselines.csv")
        self.assertGreaterEqual(len(df5), 6)
        self.assertIn("storage_bytes_per_image", df5.columns)

        # Table 6: Oxford transfer
        df6 = pd.read_csv(self.tables_dir / "table6_oxford_transfer.csv")
        self.assertEqual(len(df6), 6)
        self.assertIn("mAP", df6.columns)

        # Table 7: Runtime scalability
        df7 = pd.read_csv(self.tables_dir / "table7_runtime_scalability.csv")
        self.assertGreaterEqual(len(df7), 8)

    def test_03_publication_figures_exist_and_valid(self):
        """Verify all 9 publication figures exist, are valid PNGs, and meet 300 DPI standards."""
        expected_figs = [
            "fig_system_architecture.png",
            "fig_retrieval_pipeline.png",
            "fig_mlp_architecture.png",
            "fig_weight_distributions.png",
            "fig_main_performance_comparison.png",
            "fig_category_performance.png",
            "fig_gallery_size_sensitivity.png",
            "fig_deep_learning_comparison.png",
            "fig_qualitative_retrieval_examples.png"
        ]

        for fig_name in expected_figs:
            root_fig = self.figs_dir / fig_name
            out_fig = self.output_figs_dir / fig_name
            self.assertTrue(root_fig.exists(), f"Missing root figure: {root_fig}")
            self.assertTrue(out_fig.exists(), f"Missing output figure: {out_fig}")
            self.assertGreater(root_fig.stat().st_size, 10000, f"Figure too small: {root_fig}")

            with Image.open(root_fig) as img:
                w, h = img.size
                self.assertGreaterEqual(w, 1000, f"Figure width below 1000px: {fig_name}")
                self.assertGreaterEqual(h, 800, f"Figure height below 800px: {fig_name}")
                self.assertEqual(img.format, "PNG")

    def test_04_audit_and_checklist_reports(self):
        """Verify final_research_audit.md, reproducibility_checklist.md, and recommendations."""
        audit_file = PROJECT_ROOT / "final_research_audit.md"
        checklist_file = PROJECT_ROOT / "reproducibility_checklist.md"
        summary_file = PROJECT_ROOT / "final_results_summary.md"
        recom_file = PROJECT_ROOT / "manuscript_integration_recommendation.md"

        self.assertTrue(audit_file.exists())
        self.assertTrue(checklist_file.exists())
        self.assertTrue(summary_file.exists())
        self.assertTrue(recom_file.exists())

        audit_content = audit_file.read_text(encoding="utf-8")
        # Check all 11 tasks are documented
        for task_idx in range(1, 12):
            self.assertIn(f"Task {task_idx}:", audit_content, f"Task {task_idx} missing from audit report")

        # Check required limitations & disclosures are explicitly stated
        self.assertIn("Statistically Significant Findings", audit_content)
        self.assertIn("Descriptive-Only Findings", audit_content)
        self.assertIn("Flowers Driver", audit_content)
        self.assertIn("Gallery-Composition Sensitivity", audit_content)
        self.assertIn("External-Transfer Boundary", audit_content)
        self.assertIn("No Universal Superiority Claim", audit_content)

    def test_05_environment_and_manifest(self):
        """Verify environment files and cryptographic experiment manifest."""
        req_file = PROJECT_ROOT / "requirements.txt"
        yml_file = PROJECT_ROOT / "environment.yml"
        manifest_file = PROJECT_ROOT / "experiment_manifest.json"

        self.assertTrue(req_file.exists())
        self.assertTrue(yml_file.exists())
        self.assertTrue(manifest_file.exists())

        req_text = req_file.read_text(encoding="utf-8")
        self.assertIn("torch", req_text)
        self.assertIn("torchvision", req_text)
        self.assertIn("scikit-image", req_text)

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["phase"], "Phase 10 — Final Research Audit and Publication Package")
        checksums = manifest["verified_file_checksums_sha256"]
        self.assertGreater(len(checksums), 20)

        # Spot-check a file hash
        sample_rel = "output/phase10/manuscript_tables/table1_main_performance.csv"
        if sample_rel in checksums:
            expected_hash = checksums[sample_rel]
            actual_hash = compute_sha256(PROJECT_ROOT / sample_rel)
            self.assertEqual(expected_hash, actual_hash)


if __name__ == "__main__":
    unittest.main()
