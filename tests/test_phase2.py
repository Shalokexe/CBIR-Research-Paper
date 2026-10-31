"""
Automated Unit Tests for Phase 2: HSV Color & GLCM Texture Feature Extraction.
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

import config
from data.dataset_loader import load_dataset
from features.feature_extractor import (
    extract_hsv_color_features,
    extract_glcm_texture_features,
    extract_all_features,
)
from features.normalization import MinMaxNormalizer


class TestPhase2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_root = config.DATASET_PATH
        cls.gallery_paths, cls.gallery_labels, cls.query_paths, cls.query_labels = load_dataset(
            dataset_path=cls.dataset_root, seed=config.RANDOM_SEED
        )
        cls.sample_image = cls.query_paths[0]

    def test_hsv_feature_extraction_shape_and_types(self):
        """Verify HSV feature extraction produces exactly 9 float32 descriptors."""
        c_feats = extract_hsv_color_features(self.sample_image)
        self.assertIsInstance(c_feats, np.ndarray)
        self.assertEqual(c_feats.shape, (9,))
        self.assertEqual(c_feats.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(c_feats)), "HSV features must be finite (no NaN or Inf).")

        # Check bounds: H mean ∈ [0, 1], S mean ∈ [0, 1], V mean ∈ [0, 1]
        h_mean, h_std, h_skew, s_mean, s_std, s_skew, v_mean, v_std, v_skew = c_feats
        self.assertGreaterEqual(h_mean, 0.0)
        self.assertLessEqual(h_mean, 1.0)
        self.assertGreaterEqual(s_mean, 0.0)
        self.assertLessEqual(s_mean, 1.0)
        self.assertGreaterEqual(v_mean, 0.0)
        self.assertLessEqual(v_mean, 1.0)
        self.assertGreaterEqual(h_std, 0.0)
        self.assertGreaterEqual(s_std, 0.0)
        self.assertGreaterEqual(v_std, 0.0)

    def test_glcm_feature_extraction_shape_and_properties(self):
        """Verify GLCM texture extraction produces exactly 4 float32 descriptors."""
        t_feats = extract_glcm_texture_features(self.sample_image)
        self.assertIsInstance(t_feats, np.ndarray)
        self.assertEqual(t_feats.shape, (4,))
        self.assertEqual(t_feats.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(t_feats)), "GLCM features must be finite.")

        contrast, correlation, energy, homogeneity = t_feats
        self.assertGreaterEqual(contrast, 0.0, "Contrast must be non-negative.")
        self.assertGreaterEqual(correlation, -1.0, "Correlation must be >= -1.0.")
        self.assertLessEqual(correlation, 1.0, "Correlation must be <= 1.0.")
        self.assertGreaterEqual(energy, 0.0, "Energy must be in [0, 1].")
        self.assertLessEqual(energy, 1.0, "Energy must be in [0, 1].")
        self.assertGreaterEqual(homogeneity, 0.0, "Homogeneity must be in [0, 1].")
        self.assertLessEqual(homogeneity, 1.0, "Homogeneity must be in [0, 1].")

    def test_constant_image_zero_variance_handling(self):
        """Verify constant/flat images produce std=0 and skewness=0 without numerical errors."""
        temp_img_path = Path("scratch_constant_test.png")
        try:
            # Create a 64x64 solid gray image
            flat_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            flat_img.save(temp_img_path)

            c_feats = extract_hsv_color_features(temp_img_path)
            self.assertEqual(c_feats.shape, (9,))
            self.assertTrue(np.all(np.isfinite(c_feats)))

            # Skewness must be safely set to 0.0 when std < 1e-6
            h_std, h_skew = c_feats[1], c_feats[2]
            s_std, s_skew = c_feats[4], c_feats[5]
            v_std, v_skew = c_feats[7], c_feats[8]

            self.assertLess(s_std, 1e-5)
            self.assertEqual(s_skew, 0.0)
            self.assertLess(v_std, 1e-5)
            self.assertEqual(v_skew, 0.0)

            t_feats = extract_glcm_texture_features(temp_img_path)
            self.assertEqual(t_feats.shape, (4,))
            self.assertTrue(np.all(np.isfinite(t_feats)))
            self.assertAlmostEqual(t_feats[0], 0.0, places=4)  # 0 contrast on flat image
            self.assertAlmostEqual(t_feats[3], 1.0, places=4)  # 1 homogeneity on flat image
        finally:
            if temp_img_path.exists():
                temp_img_path.unlink()

    def test_combined_13d_descriptor(self):
        """Verify combined feature extraction returns exactly 9+4 = 13 dimensions."""
        c_feats, t_feats = extract_all_features(self.sample_image)
        combined = np.concatenate([c_feats, t_feats])
        self.assertEqual(combined.shape, (13,))
        self.assertEqual(combined.dtype, np.float32)

    def test_extraction_determinism(self):
        """Verify feature extraction is 100% deterministic on repeated runs."""
        c1, t1 = extract_all_features(self.sample_image)
        c2, t2 = extract_all_features(self.sample_image)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(t1, t2)

    def test_gallery_only_normalizer(self):
        """Verify normalizer fits on gallery only and does not leak query distributions."""
        dummy_gallery = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
            [50.0, 60.0]
        ], dtype=np.float32)

        dummy_queries = np.array([
            [0.0, 20.0],   # Below gallery min
            [60.0, 80.0]   # Above gallery max
        ], dtype=np.float32)

        normalizer = MinMaxNormalizer()
        normalizer.fit(dummy_gallery)

        # Min/max should reflect ONLY the gallery
        np.testing.assert_allclose(normalizer.min_val, [10.0, 20.0])
        np.testing.assert_allclose(normalizer.max_val, [50.0, 60.0])

        norm_gallery = normalizer.transform(dummy_gallery)
        self.assertTrue(np.all(norm_gallery >= 0.0))
        self.assertTrue(np.all(norm_gallery <= 1.0))

        # Query points outside gallery range should appropriately be < 0 or > 1
        norm_queries = normalizer.transform(dummy_queries)
        self.assertLess(norm_queries[0, 0], 0.0)
        self.assertGreater(norm_queries[1, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
