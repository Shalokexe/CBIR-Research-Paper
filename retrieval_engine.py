"""
Complete CBIR Retrieval Engine Module supporting M1 through M6.
Implements:
- M1: Colour-Only
- M2: Texture-Only
- M3: Equal Feature-Level Fusion (0.5/0.5)
- M4: Fixed Feature-Level Fusion (0.7/0.3)
- M5: Query-Adaptive Distance-Level Fusion (MLP predicted weights)
- M6: Fixed Distance-Level Control (Mean predicted query weights)
"""

import time
from typing import Dict, List, Tuple, Any, Optional
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
from model.mlp_model import AdaptiveWeightMLP


class CBIRRetrievalEngine:
    """
    Modular CBIR Engine executing image retrieval and ranking across all six models (M1–M6).
    """
    def __init__(
        self,
        gallery_color: np.ndarray,
        gallery_texture: np.ndarray,
        gallery_labels: List[str] | np.ndarray,
        gallery_paths: List[str] | np.ndarray
    ):
        self.gallery_color = np.asarray(gallery_color, dtype=np.float32)
        self.gallery_texture = np.asarray(gallery_texture, dtype=np.float32)
        self.gallery_labels = np.asarray(gallery_labels)
        self.gallery_paths = np.asarray(gallery_paths)

        self.gallery_size = len(self.gallery_labels)
        if self.gallery_color.shape[0] != self.gallery_size or self.gallery_texture.shape[0] != self.gallery_size:
            raise ValueError("Gallery feature sample counts do not match label count.")

        # Precompute concatenated representations for fast baseline retrieval
        self.gallery_m3 = feature_level_fusion(
            self.gallery_color, self.gallery_texture, config.M3_COLOR_WEIGHT, config.M3_TEXTURE_WEIGHT
        )
        self.gallery_m4 = feature_level_fusion(
            self.gallery_color, self.gallery_texture, config.M4_COLOR_WEIGHT, config.M4_TEXTURE_WEIGHT
        )

    def retrieve_single(
        self,
        query_color: np.ndarray,
        query_texture: np.ndarray,
        model_name: str,
        weights: Optional[np.ndarray | List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executes retrieval for a single query image against the gallery.

        Args:
            query_color: 1D array of shape (9,).
            query_texture: 1D array of shape (4,).
            model_name: 'M1', 'M2', 'M3', 'M4', 'M5', or 'M6'.
            weights: Optional weight pair [w_C, w_T] required for M5 and M6.

        Returns:
            ranked_indices: 1D array of sorted gallery indices.
            sorted_distances: 1D array of sorted Euclidean distances.
            elapsed_time: Distance computation and ranking time in seconds.
        """
        t0 = time.perf_counter()

        if model_name == "M1":
            # Colour only: Euclidean distance in 9-D space
            dists = compute_color_distance(query_color, self.gallery_color)

        elif model_name == "M2":
            # Texture only: Euclidean distance in 4-D space
            dists = compute_texture_distance(query_texture, self.gallery_texture)

        elif model_name == "M3":
            # Equal feature-level fusion: 0.5 * C and 0.5 * T concatenated
            q_m3 = feature_level_fusion(query_color, query_texture, config.M3_COLOR_WEIGHT, config.M3_TEXTURE_WEIGHT)
            diff = self.gallery_m3 - q_m3
            dists = np.sqrt(np.maximum(np.sum(diff ** 2, axis=1), 0.0))

        elif model_name == "M4":
            # Fixed 0.7/0.3 feature-level fusion
            q_m4 = feature_level_fusion(query_color, query_texture, config.M4_COLOR_WEIGHT, config.M4_TEXTURE_WEIGHT)
            diff = self.gallery_m4 - q_m4
            dists = np.sqrt(np.maximum(np.sum(diff ** 2, axis=1), 0.0))

        elif model_name in ("M5", "M6"):
            if weights is None:
                raise ValueError(f"Model {model_name} requires weights [w_C, w_T].")
            w = validate_fusion_weights(weights)

            # Compute squared distances independently
            dc_sq = compute_squared_color_distance(query_color, self.gallery_color)
            dt_sq = compute_squared_texture_distance(query_texture, self.gallery_texture)

            # Adaptive distance fusion: D = sqrt(w_C * D_C^2 + w_T * D_T^2)
            dists = adaptive_distance_fusion(dc_sq, dt_sq, wc=w[0], wt=w[1])

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        ranked_indices, sorted_distances = rank_distances(dists)
        t1 = time.perf_counter()

        return ranked_indices, sorted_distances, (t1 - t0)

    def retrieve_batch(
        self,
        query_color: np.ndarray,
        query_texture: np.ndarray,
        model_name: str,
        weights_by_query: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Executes retrieval for a batch of queries.

        Args:
            query_color: 2D array of shape (Q, 9).
            query_texture: 2D array of shape (Q, 4).
            model_name: Model identifier ('M1' to 'M6').
            weights_by_query: Array of shape (Q, 2) weights for M5 or M6.

        Returns:
            all_ranked_indices: Array of shape (Q, G).
            all_sorted_distances: Array of shape (Q, G).
            latencies: List of execution times per query in seconds.
        """
        num_queries = query_color.shape[0]
        all_rankings = []
        all_distances = []
        latencies = []

        for q_idx in range(num_queries):
            q_c = query_color[q_idx]
            q_t = query_texture[q_idx]
            w = weights_by_query[q_idx] if weights_by_query is not None else None

            ranked_idx, sorted_dists, elapsed = self.retrieve_single(
                query_color=q_c,
                query_texture=q_t,
                model_name=model_name,
                weights=w
            )
            all_rankings.append(ranked_idx)
            all_distances.append(sorted_dists)
            latencies.append(elapsed)

        return np.array(all_rankings), np.array(all_distances), latencies
