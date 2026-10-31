"""
Automated Unit Tests for Phase 3: Baseline Retrieval Models (M1–M4) and Evaluation Engine.
"""

import unittest
import numpy as np

import config
from retrieval.distance import (
    compute_feature_representation,
    compute_euclidean_distance,
    rank_gallery,
)
from retrieval.evaluation import (
    evaluate_single_query,
    evaluate_all_queries,
    aggregate_category_metrics,
)


class TestPhase3(unittest.TestCase):
    def setUp(self):
        # Synthetic feature vectors: 2 samples, 9 color dims, 4 texture dims
        self.color = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        ], dtype=np.float32)

        self.texture = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.5, 0.5, 0.5]
        ], dtype=np.float32)

    def test_feature_representations_and_shapes(self):
        """Verify baseline feature concatenation and shapes."""
        m1 = compute_feature_representation(self.color, self.texture, "M1")
        self.assertEqual(m1.shape, (2, 9))
        np.testing.assert_array_equal(m1, self.color)

        m2 = compute_feature_representation(self.color, self.texture, "M2")
        self.assertEqual(m2.shape, (2, 4))
        np.testing.assert_array_equal(m2, self.texture)

        m3 = compute_feature_representation(self.color, self.texture, "M3")
        self.assertEqual(m3.shape, (2, 13))
        np.testing.assert_array_equal(m3[:, :9], 0.5 * self.color)
        np.testing.assert_array_equal(m3[:, 9:], 0.5 * self.texture)

        m4 = compute_feature_representation(self.color, self.texture, "M4")
        self.assertEqual(m4.shape, (2, 13))
        np.testing.assert_array_equal(m4[:, :9], 0.7 * self.color)
        np.testing.assert_array_equal(m4[:, 9:], 0.3 * self.texture)

    def test_feature_level_squared_distance_implication(self):
        """Verify that feature-level fusion induces squared coefficients in distance space."""
        q_c = self.color[0]
        g_c = self.color[1:2]
        q_t = self.texture[0]
        g_t = self.texture[1:2]

        d_c_sq = np.sum((g_c - q_c) ** 2)
        d_t_sq = np.sum((g_t - q_t) ** 2)

        # M3 distance
        q_m3 = compute_feature_representation(q_c, q_t, "M3")
        g_m3 = compute_feature_representation(g_c, g_t, "M3")
        dist_m3 = compute_euclidean_distance(q_m3, g_m3)[0]
        expected_m3 = np.sqrt(0.25 * d_c_sq + 0.25 * d_t_sq)
        self.assertAlmostEqual(float(dist_m3), float(expected_m3), places=5)

        # M4 distance
        q_m4 = compute_feature_representation(q_c, q_t, "M4")
        g_m4 = compute_feature_representation(g_c, g_t, "M4")
        dist_m4 = compute_euclidean_distance(q_m4, g_m4)[0]
        expected_m4 = np.sqrt(0.49 * d_c_sq + 0.09 * d_t_sq)
        self.assertAlmostEqual(float(dist_m4), float(expected_m4), places=5)

    def test_euclidean_distance_and_ranking_order(self):
        """Verify ranking produces ascending distance order."""
        query = np.array([0.0, 0.0], dtype=np.float32)
        gallery = np.array([
            [3.0, 4.0],  # dist = 5.0
            [1.0, 0.0],  # dist = 1.0
            [0.0, 2.0]   # dist = 2.0
        ], dtype=np.float32)

        dists = compute_euclidean_distance(query, gallery)
        ranked_indices, sorted_dists = rank_gallery(dists)

        # Nearest is index 1 (dist 1.0), then index 2 (dist 2.0), then index 0 (dist 5.0)
        np.testing.assert_array_equal(ranked_indices, np.array([1, 2, 0]))
        np.testing.assert_allclose(sorted_dists, np.array([1.0, 2.0, 5.0]))

    def test_single_query_metrics_analytic_verification(self):
        """Verify analytical computation of P@10, R@10, unrounded F1@10, and mAP."""
        # Gallery of 20 images: 2 relevant in top 10, 10 relevant in remaining 10 -> total 12 relevant
        labels = ["target"] * 2 + ["other"] * 8 + ["target"] * 10
        ranked_indices = np.arange(20)

        metrics = evaluate_single_query(ranked_indices, labels, "target", k=10)

        # In top 10: 2 target images
        self.assertEqual(metrics["precision_k"], 0.2)
        self.assertAlmostEqual(metrics["recall_k"], 2.0 / 12.0)

        expected_f1 = 2 * (0.2 * (2.0 / 12.0)) / (0.2 + (2.0 / 12.0))
        self.assertAlmostEqual(metrics["f1_k"], expected_f1)

        # Average precision: ranks 1, 2 (precisions 1/1, 2/2) + ranks 11..20 (precisions 3/11 .. 12/20)
        expected_ap = (
            2.0 + sum((idx + 3.0) / (idx + 11.0) for idx in range(10))
        ) / 12.0
        self.assertAlmostEqual(metrics["ap"], expected_ap)

    def test_ranking_permutation_validation(self):
        """Verify evaluate_single_query rejects invalid or duplicate rankings."""
        labels = ["cat", "dog", "bird"]
        invalid_rankings = [
            np.array([0, 0, 2]),      # duplicate index
            np.array([0, 1, 3]),      # index out of range
            np.array([0, 1]),         # incomplete ranking
            np.array([[0], [1], [2]]) # 2D array
        ]
        for bad_ranking in invalid_rankings:
            with self.subTest(bad_ranking=bad_ranking):
                with self.assertRaises(ValueError):
                    evaluate_single_query(bad_ranking, labels, "cat", k=2)

    def test_query_gallery_separation_in_saved_features(self):
        """Verify that saved features cleanly separate 899 gallery from 100 queries."""
        feat_data = np.load("output/phase2/features.npz")
        partitions = feat_data["partitions"]
        gallery_mask = (partitions == "gallery")
        query_mask = (partitions == "query")

        self.assertEqual(np.sum(gallery_mask), 899)
        self.assertEqual(np.sum(query_mask), 100)
        self.assertFalse(np.any(gallery_mask & query_mask))

        # Relative paths should have zero overlap
        paths = feat_data["paths"]
        g_paths = set(paths[gallery_mask])
        q_paths = set(paths[query_mask])
        self.assertEqual(len(g_paths & q_paths), 0)


if __name__ == "__main__":
    unittest.main()
