"""
Automated Unit Tests for Phase 1: Corel-1K Dataset Protocol and Auditing.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import config
from data.dataset_loader import load_dataset
from data.duplicate_audit import (
    compute_sha256,
    compute_perceptual_hash,
    audit_dataset_duplicates,
    build_group_id_map,
)
from data.partitioning import (
    create_corrected_splits,
    group_disjoint_split,
)


class TestPhase1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_root = config.DATASET_PATH
        cls.gallery_paths, cls.gallery_labels, cls.query_paths, cls.query_labels = load_dataset(
            dataset_path=cls.dataset_root, seed=config.RANDOM_SEED
        )

    def test_dataset_discovery_and_counts(self):
        """Verify total image count and category distributions."""
        total_images = len(self.gallery_paths) + len(self.query_paths)
        self.assertEqual(total_images, 1000)
        self.assertEqual(len(self.query_paths), 100)
        self.assertEqual(len(self.gallery_paths), 900)

        # 10 queries per category
        for cat in config.EXPECTED_CATEGORIES:
            cat_queries = [p for p, l in zip(self.query_paths, self.query_labels) if l == cat]
            cat_gallery = [p for p, l in zip(self.gallery_paths, self.gallery_labels) if l == cat]
            self.assertEqual(len(cat_queries), 10, f"Category {cat} should have 10 queries.")
            self.assertEqual(len(cat_gallery), 90, f"Category {cat} should have 90 gallery images.")

    def test_query_split_determinism(self):
        """Verify that split is 100% deterministic with seed=42."""
        g2, l2, q2, ql2 = load_dataset(dataset_path=self.dataset_root, seed=config.RANDOM_SEED)
        self.assertEqual(self.gallery_paths, g2)
        self.assertEqual(self.query_paths, q2)
        self.assertEqual(self.gallery_labels, l2)
        self.assertEqual(self.query_labels, ql2)

    def test_perceptual_hash_and_sha256(self):
        """Verify hash functions return expected format and type."""
        sample_path = self.query_paths[0]
        sha = compute_sha256(sample_path)
        self.assertIsInstance(sha, str)
        self.assertEqual(len(sha), 64)

        phash = compute_perceptual_hash(sample_path)
        self.assertIsInstance(phash, int)
        self.assertGreaterEqual(phash, 0)
        self.assertLess(phash, 2**64)

    def test_known_near_duplicate_detection(self):
        """Verify that the confirmed near-duplicate pair is detected with Hamming distance 4."""
        p1 = self.dataset_root / "training_set" / "horses" / "761.jpg"
        p2 = self.dataset_root / "training_set" / "horses" / "776.jpg"

        self.assertTrue(p1.exists(), f"Missing file: {p1}")
        self.assertTrue(p2.exists(), f"Missing file: {p2}")

        h1 = compute_perceptual_hash(p1)
        h2 = compute_perceptual_hash(p2)
        dist = (h1 ^ h2).bit_count()

        self.assertEqual(dist, 4)
        self.assertEqual(f"{h1:016x}", "a565d6cf213296c3")
        self.assertEqual(f"{h2:016x}", "a165d6db2132b6c3")

    def test_corrected_splits_and_zero_leakage(self):
        """Verify 899 gallery, 100 queries, 700 train, 199 val, and strict zero leakage."""
        split_info = create_corrected_splits(
            raw_gallery_paths=self.gallery_paths,
            raw_gallery_labels=self.gallery_labels,
            query_paths=self.query_paths,
            query_labels=self.query_labels,
            dataset_root=self.dataset_root,
            query_seed=config.RANDOM_SEED,
            mlp_seed=config.MLP_SEED
        )

        gallery = set(split_info["gallery_paths"])
        queries = set(split_info["query_paths"])
        train = set(split_info["train_paths"])
        val = set(split_info["val_paths"])
        excluded = set(split_info["excluded_paths"])

        # Cardinalities
        self.assertEqual(len(gallery), 899)
        self.assertEqual(len(queries), 100)
        self.assertEqual(len(train), 700)
        self.assertEqual(len(val), 199)
        self.assertEqual(len(excluded), 1)

        # Mutual exclusivity
        self.assertFalse(gallery & queries, "Gallery and query sets must be disjoint.")
        self.assertFalse(train & val, "Train and validation sets must be disjoint.")
        self.assertEqual(train | val, gallery, "Train and validation must partition the gallery.")
        self.assertFalse(train & queries, "Train and query sets must be disjoint.")
        self.assertFalse(val & queries, "Validation and query sets must be disjoint.")
        self.assertFalse(gallery & excluded, "Excluded duplicate must not be in gallery.")
        self.assertFalse(queries & excluded, "Excluded duplicate must not be in queries.")

        # Check specific excluded file
        excluded_rel = [p.relative_to(self.dataset_root).as_posix() for p in excluded]
        self.assertEqual(excluded_rel, ["training_set/horses/761.jpg"])

        # Check retained query
        query_rel = [p.relative_to(self.dataset_root).as_posix() for p in queries]
        self.assertIn("training_set/horses/776.jpg", query_rel)

    def test_group_disjoint_split_invariants(self):
        """Verify group-disjoint split prevents duplicate groups from crossing train/val."""
        dummy_paths = [Path(f"cat_{i:03d}.jpg") for i in range(90)]
        dummy_labels = ["cat"] * 90
        # Link images 0 and 1 into a duplicate group
        dummy_groups = {
            dummy_paths[0]: "group_1",
            dummy_paths[1]: "group_1"
        }

        train_idx, val_idx = group_disjoint_split(
            dummy_paths, dummy_labels, dummy_groups, seed=42
        )

        self.assertEqual(len(train_idx), 70)
        self.assertEqual(len(val_idx), 20)
        self.assertFalse(set(train_idx) & set(val_idx))

        # Both linked images must be either in train or in val
        idx0_in_train = 0 in train_idx
        idx1_in_train = 1 in train_idx
        self.assertEqual(idx0_in_train, idx1_in_train, "Group members must be in the same partition.")


if __name__ == "__main__":
    unittest.main()
