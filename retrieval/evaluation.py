"""
Evaluation Metrics Engine for Content-Based Image Retrieval (CBIR).
Implements exact unrounded Precision@K, Recall@K, F1@K, and full-gallery Mean Average Precision (mAP).
"""

from typing import Dict, List, Tuple, Any
import numpy as np

import config


def evaluate_single_query(
    ranked_indices: np.ndarray,
    gallery_labels: List[str] | np.ndarray,
    query_label: str,
    k: int = config.EVAL_K
) -> Dict[str, float]:
    """
    Computes retrieval evaluation metrics for a single query against the ranked gallery.

    Mathematical Specifications:
    - Relevance Vector: rel[i] = 1 if gallery_labels[ranked_indices[i]] == query_label else 0
    - Total Relevant: N_rel = sum(rel)
    - Precision@K:
      P@K = (1 / K) * sum_{i=1}^K rel[i]
    - Recall@K:
      R@K = (1 / N_rel) * sum_{i=1}^K rel[i]  (0.0 if N_rel == 0)
    - F1-score@K:
      Calculated per-query from unrounded P@K and R@K:
      F1@K = 2 * (P@K * R@K) / (P@K + R@K)   (0.0 if P@K + R@K == 0)
    - Full-Gallery Average Precision (AP):
      Evaluated across the entire gallery ranking (all G gallery elements):
      AP = (1 / N_rel) * sum_{i=1}^G [ P(i) * rel[i] ]
      where P(i) = (sum_{j=1}^i rel[j]) / i is the precision at cut-off rank i.

    Args:
        ranked_indices: 1D array of sorted gallery indices (shape (G,)).
        gallery_labels: Category label for each gallery item (length G).
        query_label: Ground-truth category of the query.
        k: Cut-off rank K for P@K, R@K, F1@K (default: 10).

    Returns:
        Dict with keys: 'precision_k', 'recall_k', 'f1_k', 'ap', 'total_relevant', 'tp_k'.
    """
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer; got {k}")

    ranked_idx = np.asarray(ranked_indices, dtype=np.int64)
    g_labels = np.asarray(gallery_labels)

    if ranked_idx.ndim != 1:
        raise ValueError(f"Ranked indices must be 1D array; got shape {ranked_idx.shape}")
    if len(ranked_idx) != len(g_labels):
        raise ValueError(
            f"Length mismatch: {len(ranked_idx)} ranked indices vs {len(g_labels)} gallery labels"
        )
    if k > len(ranked_idx):
        raise ValueError(f"k={k} cannot exceed gallery size={len(ranked_idx)}")

    # Invariant: ranked_idx must be a valid permutation of range(len(gallery))
    if not np.array_equal(np.sort(ranked_idx), np.arange(len(g_labels))):
        raise ValueError("Ranked indices must be a valid permutation of gallery indices 0..G-1.")

    # Relevance vector (1 for ground-truth match, 0 otherwise)
    ranked_labels = g_labels[ranked_idx]
    rel = (ranked_labels == query_label).astype(np.int32)
    total_relevant = int(np.sum(rel))

    # Top-K True Positives
    tp_k = int(np.sum(rel[:k]))
    precision_k = float(tp_k / k)
    recall_k = float(tp_k / total_relevant) if total_relevant > 0 else 0.0

    # Unrounded F1-score
    if (precision_k + recall_k) > 0.0:
        f1_k = float(2.0 * (precision_k * recall_k) / (precision_k + recall_k))
    else:
        f1_k = 0.0

    # Average Precision over the entire gallery
    ranks = np.arange(1, len(rel) + 1, dtype=np.float64)
    cum_rel = np.cumsum(rel, dtype=np.float64)
    prec_at_ranks = cum_rel / ranks

    if total_relevant > 0:
        ap = float(np.sum(prec_at_ranks * rel) / total_relevant)
    else:
        ap = 0.0

    return {
        "precision_k": precision_k,
        "recall_k": recall_k,
        "f1_k": f1_k,
        "ap": ap,
        "total_relevant": total_relevant,
        "tp_k": tp_k
    }


def evaluate_all_queries(
    all_ranked_indices: List[np.ndarray],
    gallery_labels: List[str] | np.ndarray,
    query_labels: List[str] | np.ndarray,
    query_paths: List[str],
    latencies: List[float],
    model_name: str,
    k: int = config.EVAL_K
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Evaluates all queries for a given model, returning per-query rows and aggregated means.
    """
    if len(all_ranked_indices) != len(query_labels):
        raise ValueError("Query rankings count must match query labels count.")
    if len(latencies) != len(query_labels):
        raise ValueError("Latencies count must match query count.")

    per_query_rows: List[Dict[str, Any]] = []
    precisions = []
    recalls = []
    f1s = []
    aps = []

    for q_idx in range(len(query_labels)):
        q_path = query_paths[q_idx]
        q_lbl = query_labels[q_idx]
        ranked_idx = all_ranked_indices[q_idx]
        latency = latencies[q_idx]

        metrics = evaluate_single_query(ranked_idx, gallery_labels, q_lbl, k=k)
        precisions.append(metrics["precision_k"])
        recalls.append(metrics["recall_k"])
        f1s.append(metrics["f1_k"])
        aps.append(metrics["ap"])

        per_query_rows.append({
            "query_index": q_idx,
            "relative_path": q_path,
            "category": q_lbl,
            "model": model_name,
            "precision_at_k": metrics["precision_k"],
            "recall_at_k": metrics["recall_k"],
            "f1_at_k": metrics["f1_k"],
            "average_precision": metrics["ap"],
            "ranking_latency_seconds": latency
        })

    aggregate_metrics = {
        "model": model_name,
        "query_count": len(query_labels),
        "mean_precision_at_10": float(np.mean(precisions)),
        "mean_recall_at_10": float(np.mean(recalls)),
        "mean_f1_at_10": float(np.mean(f1s)),
        "mean_average_precision": float(np.mean(aps)),
        "mean_ranking_latency_seconds": float(np.mean(latencies))
    }

    return per_query_rows, aggregate_metrics


def aggregate_category_metrics(per_query_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregates query-level metrics by semantic category.
    """
    category_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in per_query_rows:
        key = (row["category"], row["model"])
        category_map.setdefault(key, []).append(row)

    category_summary: List[Dict[str, Any]] = []
    for (category, model), rows in sorted(category_map.items()):
        precisions = [r["precision_at_k"] for r in rows]
        recalls = [r["recall_at_k"] for r in rows]
        f1s = [r["f1_at_k"] for r in rows]
        aps = [r["average_precision"] for r in rows]

        category_summary.append({
            "category": category,
            "model": model,
            "query_count": len(rows),
            "mean_precision_at_10": float(np.mean(precisions)),
            "mean_recall_at_10": float(np.mean(recalls)),
            "mean_f1_at_10": float(np.mean(f1s)),
            "mean_average_precision": float(np.mean(aps))
        })

    return category_summary
