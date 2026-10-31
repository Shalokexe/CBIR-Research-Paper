"""
Distance Calculation and Feature-Level Fusion Module for CBIR Baselines (M1–M4).
Defines feature-level fusion and Euclidean distance metric ranking.
"""

from typing import Tuple
import numpy as np

import config


def compute_feature_representation(
    color_feats: np.ndarray,
    texture_feats: np.ndarray,
    model_name: str
) -> np.ndarray:
    """
    Constructs the feature representation for baseline models M1, M2, M3, M4.

    Mathematical Specification:
    - M1 (Colour-only): F = C ∈ R^9
    - M2 (Texture-only): F = T ∈ R^4
    - M3 (Equal feature-level fusion): F = [0.5 * C, 0.5 * T] ∈ R^13
      * Squared-distance implication:
        D_M3^2(q, g) = ||0.5 C_q - 0.5 C_g||_2^2 + ||0.5 T_q - 0.5 T_g||_2^2
                     = 0.25 D_C^2(q, g) + 0.25 D_T^2(q, g)
    - M4 (Fixed 0.7/0.3 feature-level fusion): F = [0.7 * C, 0.3 * T] ∈ R^13
      * Squared-distance implication:
        D_M4^2(q, g) = ||0.7 C_q - 0.7 C_g||_2^2 + ||0.3 T_q - 0.3 T_g||_2^2
                     = 0.49 D_C^2(q, g) + 0.09 D_T^2(q, g)

    Args:
        color_feats: Normalized color features of shape (N, 9) or (9,).
        texture_feats: Normalized texture features of shape (N, 4) or (4,).
        model_name: 'M1', 'M2', 'M3', or 'M4'.

    Returns:
        Constructed feature matrix or vector.
    """
    color = np.asarray(color_feats, dtype=np.float32)
    texture = np.asarray(texture_feats, dtype=np.float32)

    is_1d = (color.ndim == 1)
    if is_1d:
        color = color[np.newaxis, :]
        texture = texture[np.newaxis, :]

    if color.ndim != 2 or color.shape[1] != config.COLOR_FEATURE_DIM:
        raise ValueError(f"Color features must have shape (N, {config.COLOR_FEATURE_DIM}); got {color.shape}")
    if texture.ndim != 2 or texture.shape[1] != config.TEXTURE_FEATURE_DIM:
        raise ValueError(f"Texture features must have shape (N, {config.TEXTURE_FEATURE_DIM}); got {texture.shape}")
    if color.shape[0] != texture.shape[0]:
        raise ValueError(f"Color ({color.shape[0]}) and texture ({texture.shape[0]}) sample counts must match.")

    if model_name == "M1":
        repr_out = color
    elif model_name == "M2":
        repr_out = texture
    elif model_name == "M3":
        repr_out = np.hstack([config.M3_COLOR_WEIGHT * color, config.M3_TEXTURE_WEIGHT * texture])
    elif model_name == "M4":
        repr_out = np.hstack([config.M4_COLOR_WEIGHT * color, config.M4_TEXTURE_WEIGHT * texture])
    else:
        raise ValueError(f"Unknown baseline model name: {model_name}. Supported models are M1, M2, M3, M4.")

    return repr_out[0] if is_1d else repr_out


def compute_euclidean_distance(
    query_repr: np.ndarray,
    gallery_repr: np.ndarray
) -> np.ndarray:
    """
    Computes Euclidean distance between a single query representation and all gallery representations.
    D(q, g) = sqrt(sum((q - g)^2))

    Args:
        query_repr: 1D array of shape (D,).
        gallery_repr: 2D array of shape (G, D).

    Returns:
        np.ndarray: 1D array of shape (G,) containing distances to each gallery item.
    """
    q = np.asarray(query_repr, dtype=np.float32)
    g = np.asarray(gallery_repr, dtype=np.float32)

    if q.ndim != 1:
        raise ValueError(f"Query representation must be 1D array of shape (D,); got shape {q.shape}")
    if g.ndim != 2:
        raise ValueError(f"Gallery representation must be 2D array of shape (G, D); got shape {g.shape}")
    if g.shape[1] != q.shape[0]:
        raise ValueError(f"Dimension mismatch: query has dimension {q.shape[0]} but gallery has {g.shape[1]}")

    diff = g - q
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    return dists


def rank_gallery(distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ranks gallery images in ascending order of distance (smallest distance first).

    Args:
        distances: 1D array of shape (G,) distances.

    Returns:
        ranked_indices (np.ndarray): Indices of gallery items sorted ascending by distance.
        sorted_distances (np.ndarray): Distances in ascending order.
    """
    dists = np.asarray(distances, dtype=np.float32)
    if dists.ndim != 1:
        raise ValueError(f"Distances must be 1D array; got shape {dists.shape}")
    if not np.all(np.isfinite(dists)):
        raise ValueError("Distances must be finite (no NaN or Inf).")

    ranked_indices = np.argsort(dists)
    return ranked_indices, dists[ranked_indices]
