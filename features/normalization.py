"""
Gallery-Fitted Min-Max Feature Normalization Module.
Strictly ensures normalization transformations are fitted on gallery representations only,
preventing any query-feature information leakage.
"""

from typing import Dict, List, Any
import numpy as np


class MinMaxNormalizer:
    """
    Min-Max feature normalizer:
    x_norm = (x - x_min) / (x_max - x_min)

    Methodological Invariant:
    `fit` MUST be invoked using permitted gallery samples only.
    Query features and excluded duplicate images are strictly prohibited
    from influencing `min_val` or `max_val`.
    """
    def __init__(self, min_val: np.ndarray | None = None, max_val: np.ndarray | None = None):
        self.min_val = np.asarray(min_val, dtype=np.float32) if min_val is not None else None
        self.max_val = np.asarray(max_val, dtype=np.float32) if max_val is not None else None

    def fit(self, features: np.ndarray) -> "MinMaxNormalizer":
        """
        Fits column-wise minimum and maximum bounds on training/gallery features.

        Args:
            features: 2D array of shape (N, D).
        """
        features_arr = np.asarray(features, dtype=np.float32)
        if features_arr.ndim != 2:
            raise ValueError(f"Features must be 2D array of shape (N, D); got ndim={features_arr.ndim}")
        if features_arr.shape[0] == 0:
            raise ValueError("Cannot fit normalizer on empty feature array.")

        self.min_val = np.min(features_arr, axis=0)
        self.max_val = np.max(features_arr, axis=0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transforms features using pre-fitted bounds.

        Args:
            features: 2D array of shape (M, D).

        Returns:
            np.ndarray: Normalized features of shape (M, D) with values typically in [0.0, 1.0].
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer must be fitted before calling transform().")

        features_arr = np.asarray(features, dtype=np.float32)
        if features_arr.ndim != 2:
            raise ValueError(f"Features must be 2D array; got ndim={features_arr.ndim}")
        if features_arr.shape[1] != self.min_val.shape[0]:
            raise ValueError(
                f"Feature dimension {features_arr.shape[1]} does not match "
                f"fitted dimension {self.min_val.shape[0]}"
            )

        range_val = self.max_val - self.min_val
        # Replace zero or near-zero range with 1.0 to prevent division by zero
        range_val = np.where(range_val < 1e-8, 1.0, range_val)

        normalized = (features_arr - self.min_val) / range_val
        return np.asarray(normalized, dtype=np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fits on provided features and returns transformed features."""
        return self.fit(features).transform(features)

    def to_dict(self) -> Dict[str, List[float]]:
        """Exports fitted parameters to a serializable dictionary."""
        if self.min_val is None or self.max_val is None:
            return {"min": [], "max": []}
        return {
            "min": [float(v) for v in self.min_val],
            "max": [float(v) for v in self.max_val]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[float]]) -> "MinMaxNormalizer":
        """Instantiates a normalizer from serialized parameter dictionary."""
        return cls(min_val=np.array(data["min"]), max_val=np.array(data["max"]))
