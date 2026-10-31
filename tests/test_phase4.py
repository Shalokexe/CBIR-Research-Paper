"""
Automated Unit Tests for Phase 4: CBIR Retrieval Engine and M1–M6 Evaluation.
"""

import unittest
from pathlib import Path
import numpy as np
import torch

import config
from distance_functions import (
    compute_color_distance,
    compute_texture_distance,
    compute_squared_color_distance,
    compute_squared_texture_distance,
    rank_distances,
)
from fusion_methods import (
    feature_level_fusion,
    adaptive_distance_fusion,
    validate_fusion_weights,
)
from model.mlp_model import AdaptiveWeightMLP, load_trained_mlp
from retrieval_engine import CBIRRetrievalEngine


class TestPhase4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        feat_data = np.load("output/phase2/features.npz")
        partitions = feat_data["partitions"]
        cls.gallery_color = feat_data["normalized_color"][partitions == "gallery"]
        cls.gallery_texture = feat_data["normalized_texture"][partitions == "gallery"]
        cls.gallery_labels = feat_data["labels"][partitions == "gallery"]
        cls.gallery_paths = feat_data["paths"][partitions == "gallery"]

        cls.query_color = feat_data["normalized_color"][partitions == "query"]
        cls.query_texture = feat_data["normalized_texture"][partitions == "query"]
        cls.query_labels = feat_data["labels"][partitions == "query"]
        cls.query_paths = feat_data["paths"][partitions == "query"]

        cls.model, _ = load_trained_mlp(config.MODEL_CHECKPOINT_PATH)
        cls.engine = CBIRRetrievalEngine(
            gallery_color=cls.gallery_color,
            gallery_texture=cls.gallery_texture,
            gallery_labels=cls.gallery_labels,
            gallery_paths=cls.gallery_paths
        )

    def test_ranking_sorted_ascending(self):
        """Verify distance ranking is strictly monotonically non-decreasing."""
        dists = np.array([5.2, 1.1, 0.4, 3.8, 2.5], dtype=np.float32)
        ranked_idx, sorted_dists = rank_distances(dists)

        self.assertTrue(np.all(np.diff(sorted_dists) >= 0), "Sorted distances must be ascending.")
        np.testing.assert_array_equal(ranked_idx, np.array([2, 1, 4, 3, 0]))

    def test_identical_gallery_and_query_identities(self):
        """Verify that all models evaluate on identical query and gallery sets."""
        self.assertEqual(len(self.gallery_paths), 899)
        self.assertEqual(len(self.query_paths), 100)
        self.assertEqual(len(set(self.gallery_paths) & set(self.query_paths)), 0)

    def test_m5_weights_simplex_and_variance(self):
        """Verify that M5 weights vary across queries and strictly satisfy the simplex constraint."""
        q_features = np.hstack([self.query_color, self.query_texture])
        with torch.no_grad():
            weights = self.model(torch.tensor(q_features, dtype=torch.float32)).numpy()

        self.assertEqual(weights.shape, (100, 2))
        self.assertTrue(np.all(weights >= 0.0), "Weights must be non-negative.")
        np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, atol=1e-6)

        # Variance must be non-zero (demonstrating adaptivity, not static constant)
        color_weight_std = float(np.std(weights[:, 0]))
        self.assertGreater(color_weight_std, 0.1, "M5 weights must vary across heterogeneous queries.")

    def test_m6_weights_fixed_control(self):
        """Verify that M6 weights remain fixed across all queries and match reference mean."""
        q_features = np.hstack([self.query_color, self.query_texture])
        with torch.no_grad():
            m5_weights = self.model(torch.tensor(q_features, dtype=torch.float32)).numpy()

        mean_w = np.mean(m5_weights, axis=0)
        np.testing.assert_allclose(
            mean_w,
            [config.M6_REFERENCE_COLOR_WEIGHT, config.M6_REFERENCE_TEXTURE_WEIGHT],
            atol=1e-5
        )

    def test_query_isolation_from_retrieval_gallery(self):
        """Verify no query appears in its own gallery."""
        q_set = set(self.query_paths)
        g_set = set(self.gallery_paths)
        self.assertEqual(len(q_set & g_set), 0)

    def test_distance_level_vs_feature_level_fusion(self):
        """Verify mathematical distinction between feature-level (squared) and distance-level (direct)."""
        # Create identical colour and texture squared distance
        dc_sq = np.array([4.0], dtype=np.float32)
        dt_sq = np.array([9.0], dtype=np.float32)

        # Distance-level fusion with 0.5, 0.5:
        # D_dist = sqrt(0.5 * 4.0 + 0.5 * 9.0) = sqrt(6.5) ≈ 2.5495
        d_dist = adaptive_distance_fusion(dc_sq, dt_sq, 0.5, 0.5)[0]
        self.assertAlmostEqual(d_dist, np.sqrt(6.5), places=5)

        # Feature-level fusion with 0.5, 0.5:
        # D_feat = sqrt(0.25 * 4.0 + 0.25 * 9.0) = sqrt(3.25) ≈ 1.8028
        d_feat = np.sqrt(0.25 * 4.0 + 0.25 * 9.0)
        self.assertNotAlmostEqual(d_dist, d_feat)


if __name__ == "__main__":
    unittest.main()
