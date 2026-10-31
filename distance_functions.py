"""
Distance Functions Module for CBIR Retrieval Engine.
Provides independent, numerically stable distance metrics for colour and texture modalities.

Methodological Documentation:
1. Distance Formula:
   - For an individual modality with normalized feature vectors q, g ∈ R^D:
     Euclidean Distance: D(q, g) = ||q - g||_2 = sqrt( sum_{i=1}^D (q_i - g_i)^2 )
     Squared Euclidean Distance: D^2(q, g) = sum_{i=1}^D (q_i - g_i)^2
2. Feature Scaling & Normalization:
   - Input features must be pre-normalized using the frozen gallery-fitted Min-Max scaler:
     x_hat = (x - x_min) / (x_max - x_min)
   - Both colour (9-D) and texture (4-D) are normalized independently before entering this module.
   - NO hidden or silent secondary normalization is performed during distance calculation.
3. Numerical Stability:
   - Squared differences are clamped to non-negative values before square-root: max(sum((q - g)^2), 0.0).
   - Precision is maintained in float32 / float64 without overflow or underflow.
4. Ranking Convention:
   - Ascending distance: images with smaller Euclidean distance are ranked first (index 0 is nearest / top-1).
"""

from typing import Tuple
import numpy as np


def compute_squared_color_distance(
    query_color: np.ndarray,
    gallery_color: np.ndarray
) -> np.ndarray:
    """
    Computes squared Euclidean distance in the normalized 9-dimensional colour space:
    D_C^2(q, g) = sum_{i=1}^9 (C_{q,i} - C_{g,i})^2

    Args:
        query_color: 1D array of shape (9,) or 2D array of shape (1, 9).
        gallery_color: 2D array of shape (G, 9).

    Returns:
        np.ndarray: 1D array of shape (G,) containing squared colour distances.
    """
    q = np.asarray(query_color, dtype=np.float32).ravel()
    g = np.asarray(gallery_color, dtype=np.float32)

    if q.ndim != 1 or q.shape[0] != 9:
        raise ValueError(f"query_color must have shape (9,); got shape {q.shape}")
    if g.ndim != 2 or g.shape[1] != 9:
        raise ValueError(f"gallery_color must have shape (G, 9); got shape {g.shape}")

    diff = g - q
    dc_sq = np.sum(diff ** 2, axis=1)
    return np.maximum(dc_sq, 0.0)


def compute_color_distance(
    query_color: np.ndarray,
    gallery_color: np.ndarray
) -> np.ndarray:
    """
    Computes Euclidean distance in the normalized colour space:
    D_C(q, g) = sqrt( D_C^2(q, g) )
    """
    dc_sq = compute_squared_color_distance(query_color, gallery_color)
    return np.sqrt(dc_sq)


def compute_squared_texture_distance(
    query_texture: np.ndarray,
    gallery_texture: np.ndarray
) -> np.ndarray:
    """
    Computes squared Euclidean distance in the normalized 4-dimensional texture space:
    D_T^2(q, g) = sum_{j=1}^4 (T_{q,j} - T_{g,j})^2

    Args:
        query_texture: 1D array of shape (4,) or 2D array of shape (1, 4).
        gallery_texture: 2D array of shape (G, 4).

    Returns:
        np.ndarray: 1D array of shape (G,) containing squared texture distances.
    """
    q = np.asarray(query_texture, dtype=np.float32).ravel()
    g = np.asarray(gallery_texture, dtype=np.float32)

    if q.ndim != 1 or q.shape[0] != 4:
        raise ValueError(f"query_texture must have shape (4,); got shape {q.shape}")
    if g.ndim != 2 or g.shape[1] != 4:
        raise ValueError(f"gallery_texture must have shape (G, 4); got shape {g.shape}")

    diff = g - q
    dt_sq = np.sum(diff ** 2, axis=1)
    return np.maximum(dt_sq, 0.0)


def compute_texture_distance(
    query_texture: np.ndarray,
    gallery_texture: np.ndarray
) -> np.ndarray:
    """
    Computes Euclidean distance in the normalized texture space:
    D_T(q, g) = sqrt( D_T^2(q, g) )
    """
    dt_sq = compute_squared_texture_distance(query_texture, gallery_texture)
    return np.sqrt(dt_sq)


def rank_distances(distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sorts distances in ascending order according to the standard CBIR convention.

    Args:
        distances: 1D array of shape (G,).

    Returns:
        ranked_indices: Permutation of [0, G-1] sorted by ascending distance.
        sorted_distances: Distances in ascending order.
    """
    d = np.asarray(distances, dtype=np.float32)
    if d.ndim != 1:
        raise ValueError(f"distances must be a 1D array; got shape {d.shape}")
    if not np.all(np.isfinite(d)):
        raise ValueError("Non-finite distance values detected.")

    ranked_indices = np.argsort(d)
    return ranked_indices, d[ranked_indices]
