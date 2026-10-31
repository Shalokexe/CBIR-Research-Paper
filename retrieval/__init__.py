"""
Retrieval and Evaluation Package for ICAIN 2026 Corel-1K CBIR Implementation.
"""

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

__all__ = [
    "compute_feature_representation",
    "compute_euclidean_distance",
    "rank_gallery",
    "evaluate_single_query",
    "evaluate_all_queries",
    "aggregate_category_metrics",
]
