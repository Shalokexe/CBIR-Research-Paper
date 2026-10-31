"""
Fusion Methods Module for CBIR Retrieval Engine.
Implements:
1. Feature-Level Fusion (M3, M4):
   Concatenates scaled feature representations before distance calculation:
   F = [w_C * C, w_T * T]
   Inducing: D^2(q, g) = w_C^2 * D_C^2(q, g) + w_T^2 * D_T^2(q, g)
2. Distance-Level Adaptive Fusion (M5, M6):
   Directly weights the squared Euclidean distances in each visual modality:
   D^2(q, g) = w_C * D_C^2(q, g) + w_T * D_T^2(q, g)
   where w_C, w_T >= 0 and w_C + w_T = 1.0.
"""

from typing import Tuple
import numpy as np

import config


def validate_fusion_weights(weights: np.ndarray | list | tuple, atol: float = 1e-6) -> np.ndarray:
    """
    Validates that fusion weights satisfy the simplex constraint:
    w_C >= 0, w_T >= 0, w_C + w_T = 1.0 (within atol).

    Args:
        weights: Array of shape (2,) or (N, 2).
        atol: Absolute tolerance for sum-to-one check.

    Returns:
        np.ndarray: Validated weights array.
    """
    w = np.asarray(weights, dtype=np.float32)

    if w.ndim not in (1, 2) or w.shape[-1] != 2:
        raise ValueError(f"Fusion weights must have shape (2,) or (N, 2); got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("Fusion weights must be finite (no NaN or Inf).")
    if np.any(w < -1e-6):
        raise ValueError(f"Fusion weights must be non-negative; minimum observed is {np.min(w)}")

    sums = np.sum(w, axis=-1)
    if not np.allclose(sums, 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"Fusion weights must sum to 1.0 within atol={atol}; got sums {sums}")

    return w


def feature_level_fusion(
    color_feats: np.ndarray,
    texture_feats: np.ndarray,
    wc: float,
    wt: float
) -> np.ndarray:
    """
    Performs feature-level concatenation: F = [wc * C, wt * T].
    Used by M3 (wc=0.5, wt=0.5) and M4 (wc=0.7, wt=0.3).

    Notice: Under Euclidean distance, this yields:
    D_feature^2 = wc^2 * D_C^2 + wt^2 * D_T^2
    """
    c = np.asarray(color_feats, dtype=np.float32)
    t = np.asarray(texture_feats, dtype=np.float32)

    is_1d = (c.ndim == 1)
    if is_1d:
        c = c[np.newaxis, :]
        t = t[np.newaxis, :]

    if c.shape[0] != t.shape[0]:
        raise ValueError(f"Sample count mismatch: color has {c.shape[0]}, texture has {t.shape[0]}")

    fused = np.hstack([float(wc) * c, float(wt) * t])
    return fused[0] if is_1d else fused


def adaptive_distance_fusion(
    dc_sq: np.ndarray,
    dt_sq: np.ndarray,
    wc: float | np.ndarray,
    wt: float | np.ndarray
) -> np.ndarray:
    """
    Computes direct distance-level fusion:
    D^2(q, g) = wc * D_C^2(q, g) + wt * D_T^2(q, g)
    D(q, g) = sqrt( D^2(q, g) )

    Used by:
    - M5: where wc(q), wt(q) are predicted dynamically by the MLP for query q.
    - M6: where wc, wt are fixed reference control coefficients (wc ≈ 0.5784, wt ≈ 0.4216).

    Args:
        dc_sq: 1D array of squared colour distances to gallery (shape (G,)).
        dt_sq: 1D array of squared texture distances to gallery (shape (G,)).
        wc: Scalar colour weight or 1D array.
        wt: Scalar texture weight or 1D array.

    Returns:
        np.ndarray: 1D array of fused Euclidean distances (shape (G,)).
    """
    dc_sq_arr = np.asarray(dc_sq, dtype=np.float32)
    dt_sq_arr = np.asarray(dt_sq, dtype=np.float32)

    if dc_sq_arr.shape != dt_sq_arr.shape:
        raise ValueError(f"Shape mismatch: dc_sq {dc_sq_arr.shape} vs dt_sq {dt_sq_arr.shape}")

    # Validate weights
    w = validate_fusion_weights([wc, wt])
    w_c, w_t = float(w[0]), float(w[1])

    fused_sq = w_c * dc_sq_arr + w_t * dt_sq_arr
    return np.sqrt(np.maximum(fused_sq, 0.0))
