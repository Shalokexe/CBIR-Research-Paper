"""
Handcrafted Feature Extractor Module for ICAIN 2026 Corel-1K CBIR Implementation.
Extracts:
1. HSV Color Features (9 dimensions):
   - Mean, standard deviation, and skewness for Hue (H), Saturation (S), and Value (V).
2. GLCM Texture Features (4 dimensions):
   - Contrast, Correlation, Energy, and Homogeneity averaged over 4 orientations (0°, 45°, 90°, 135°).
Total Handcrafted Descriptor: 13 dimensions.
"""

from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from skimage.feature import graycomatrix, graycoprops

import config


def extract_hsv_color_features(image_path: Path | str) -> np.ndarray:
    """
    Extracts 9-dimensional HSV color statistical features:
    [μ_H, σ_H, s_H, μ_S, σ_S, s_S, μ_V, σ_V, s_V]

    Methodology Specification:
    - Color Conversion: The image is loaded via PIL and converted to standard 24-bit RGB.
      Pixel values are cast to float32 and scaled to [0.0, 1.0] by dividing by 255.0.
      RGB pixels are transformed to HSV using `matplotlib.colors.rgb_to_hsv`.
    - Value Ranges:
      * H (Hue) ∈ [0.0, 1.0)
      * S (Saturation) ∈ [0.0, 1.0]
      * V (Value) ∈ [0.0, 1.0]
    - Calculation Type: Pixel-based calculation over all flattened channel pixels.
    - Moments:
      * Mean: μ = (1 / N) * Σ x_i
      * Standard Deviation: σ = sqrt((1 / N) * Σ (x_i - μ)^2)
      * Skewness: s = (1 / N) * Σ (x_i - μ)^3 / (σ^3)
    - Zero Variance Handling: If σ < 1e-6 (constant or near-constant channel),
      skewness is set to 0.0 to prevent division by zero and numerical instability.

    Args:
        image_path: Absolute or relative Path to the target image.

    Returns:
        np.ndarray: 1D array of shape (9,) and dtype float32.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        img_np = np.asarray(img_rgb, dtype=np.float32) / 255.0

    # Convert RGB [0, 1] to HSV [0, 1]
    hsv_img = rgb_to_hsv(img_np)

    features: List[float] = []
    # Channels: 0 -> Hue, 1 -> Saturation, 2 -> Value
    for ch_idx in range(3):
        channel = hsv_img[:, :, ch_idx].ravel()
        mean_val = float(np.mean(channel))
        std_val = float(np.std(channel))

        # Safe skewness calculation handling constant / zero-variance channels
        if std_val < 1e-6:
            skew_val = 0.0
        else:
            diff = channel - mean_val
            skew_val = float(np.mean(diff ** 3) / (std_val ** 3))

        features.extend([mean_val, std_val, skew_val])

    return np.array(features, dtype=np.float32)


def extract_glcm_texture_features(image_path: Path | str) -> np.ndarray:
    """
    Extracts 4-dimensional GLCM texture features:
    [Contrast, Correlation, Energy, Homogeneity]

    Methodology Specification:
    - Grayscale Conversion: The image is opened via PIL and converted to 8-bit grayscale ('L' mode).
      Pixels are represented as unsigned 8-bit integers (uint8) in [0, 255].
    - Quantization Levels: 256 gray levels (levels=256).
    - Pixel Distance: d = 1 (config.GLCM_DISTANCE).
    - Orientations: Four directions: θ ∈ {0°, 45°, 90°, 135°} corresponding to
      {0, π/4, π/2, 3π/4} radians (config.GLCM_ANGLES_DEG).
    - Symmetric: True (co-occurrence of (i, j) and (j, i) are counted equally).
    - Normalization: True (normed=True, elements represent joint probabilities summing to 1.0).
    - Orientation Averaging: For each property P ∈ {Contrast, Correlation, Energy, Homogeneity},
      the property is evaluated for all 4 co-occurrence matrices via `skimage.feature.graycoprops`,
      and the arithmetic mean over the four orientations is computed:
      P_avg = (1 / 4) * Σ_{θ} P(GLCM_θ).

    Args:
        image_path: Absolute or relative Path to the target image.

    Returns:
        np.ndarray: 1D array of shape (4,) and dtype float32.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    with Image.open(path) as img:
        img_gray = img.convert("L")
        img_np = np.array(img_gray, dtype=np.uint8, copy=True)

    # Convert orientation angles to radians
    angles_rad = [np.deg2rad(a) for a in config.GLCM_ANGLES_DEG]

    # Compute 256-level symmetric normalized Gray-Level Co-occurrence Matrix
    glcm = graycomatrix(
        img_np,
        distances=[config.GLCM_DISTANCE],
        angles=angles_rad,
        levels=256,
        symmetric=True,
        normed=True
    )

    # Extract properties and compute directional average across orientations
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))

    return np.array([contrast, correlation, energy, homogeneity], dtype=np.float32)


def extract_all_features(image_path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts both HSV color (9-D) and GLCM texture (4-D) feature descriptors for an image.

    Args:
        image_path: Path to target image.

    Returns:
        Tuple of (color_features [9-D], texture_features [4-D]).
    """
    color_feats = extract_hsv_color_features(image_path)
    texture_feats = extract_glcm_texture_features(image_path)
    return color_feats, texture_feats
