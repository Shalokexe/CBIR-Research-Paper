"""
Deterministic Dataset Partitioning and Leakage-Prevention Module for Corel-1K.
Produces verified, group-disjoint splits with near-duplicate exclusion:
- Gallery: 899 images (700 mlp_train, 199 mlp_validation)
- Held-out queries: 100 images
- Excluded near-duplicate: 1 image (training_set/horses/761.jpg)
"""

import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import numpy as np

import config
from data.duplicate_audit import (
    compute_sha256,
    compute_perceptual_hash,
    audit_dataset_duplicates,
    build_group_id_map,
)


def _select_groups_with_total(
    groups: List[Tuple[Any, List[int]]],
    target: int,
    rng: np.random.Generator
) -> Set[Any]:
    """
    Selects a subset of groups whose combined sizes sum to target.
    Uses dynamic programming subset-sum for exact target matching.
    """
    shuffled = list(groups)
    rng.shuffle(shuffled)
    reachable: Dict[int, List[Any]] = {0: []}

    for group_id, members in shuffled:
        size = len(members)
        for current_total, selected in list(reachable.items())[::-1]:
            next_total = current_total + size
            if next_total <= target and next_total not in reachable:
                reachable[next_total] = selected + [group_id]

    if target in reachable:
        return set(reachable[target])

    # Fallback greedy accumulator if exact sum not possible
    selected_set: Set[Any] = set()
    count = 0
    for group_id, members in shuffled:
        if count + len(members) <= target:
            selected_set.add(group_id)
            count += len(members)
    return selected_set


def group_disjoint_split(
    gallery_paths: List[Path],
    gallery_labels: List[str],
    group_ids: Dict[Path, str],
    seed: int = config.MLP_SEED
) -> Tuple[List[int], List[int]]:
    """
    Partitions the gallery into train and validation indices while guaranteeing
    that no duplicate group crosses the train/validation boundary.

    Args:
        gallery_paths: List of gallery image paths.
        gallery_labels: Category label for each gallery image.
        group_ids: Mapping from image path to duplicate group ID.
        seed: Random seed for partition selection.

    Returns:
        train_indices: Sorted list of indices assigned to MLP training.
        validation_indices: Sorted list of indices assigned to MLP validation.
    """
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    validation_indices: List[int] = []
    categories = sorted(set(gallery_labels))

    for category in categories:
        indices = [
            idx for idx, label in enumerate(gallery_labels)
            if label == category
        ]
        groups: Dict[str, List[int]] = {}
        for idx in indices:
            path = gallery_paths[idx]
            gid = group_ids.get(path, str(path))
            groups.setdefault(gid, []).append(idx)

        group_items = list(groups.items())
        target_validation = min(
            config.INTERNAL_VAL_PER_CAT,
            max(1, len(indices) - config.INTERNAL_TRAIN_PER_CAT),
        )

        selected_val_groups = _select_groups_with_total(group_items, target_validation, rng)
        selected_val = [
            idx for gid in selected_val_groups
            for idx in groups[gid]
        ]
        selected_train = [
            idx for gid, members in groups.items()
            if gid not in selected_val_groups
            for idx in members
        ]

        if len(selected_val) != target_validation or not selected_train:
            raise ValueError(
                f"Category {category!r}: could not form disjoint partition with "
                f"target validation {target_validation}. Train: {len(selected_train)}, Val: {len(selected_val)}."
            )

        train_indices.extend(selected_train)
        validation_indices.extend(selected_val)

    train_indices.sort()
    validation_indices.sort()

    # Invariant assertions
    if set(train_indices) & set(validation_indices):
        raise AssertionError("Train and validation index sets overlap!")
    if set(train_indices) | set(validation_indices) != set(range(len(gallery_paths))):
        raise AssertionError("Train and validation sets do not cover all gallery images!")

    train_groups = {group_ids.get(gallery_paths[i], str(gallery_paths[i])) for i in train_indices}
    val_groups = {group_ids.get(gallery_paths[i], str(gallery_paths[i])) for i in validation_indices}
    crossing_groups = train_groups & val_groups
    if crossing_groups:
        raise ValueError(f"Duplicate groups cross train/validation boundary: {crossing_groups}")

    return train_indices, validation_indices


def create_corrected_splits(
    raw_gallery_paths: List[Path],
    raw_gallery_labels: List[str],
    query_paths: List[Path],
    query_labels: List[str],
    dataset_root: Path | None = None,
    query_seed: int = config.RANDOM_SEED,
    mlp_seed: int = config.MLP_SEED
) -> Dict[str, Any]:
    """
    Executes the duplicate audit, excludes confirmed near-duplicate gallery images,
    and creates the verified Corel-1K split structure.

    Returns:
        Dictionary containing verified partitions, paths, labels, and audit details.
    """
    if dataset_root is None:
        dataset_root = config.DATASET_PATH

    all_paths = sorted(set(raw_gallery_paths) | set(query_paths))
    audit_results = audit_dataset_duplicates(all_paths)
    group_ids = build_group_id_map(audit_results["exact_pairs"], audit_results["near_pairs"])

    query_set = set(query_paths)
    excluded_paths: Set[Path] = set()

    for pair in audit_results["exact_pairs"] + audit_results["near_pairs"]:
        left, right = pair["left_path"], pair["right_path"]
        if left in query_set and right not in query_set:
            excluded_paths.add(right)
        elif right in query_set and left not in query_set:
            excluded_paths.add(left)

    # Filter gallery to exclude near-duplicates
    included_gallery = [
        (p, lbl) for p, lbl in zip(raw_gallery_paths, raw_gallery_labels)
        if p not in excluded_paths
    ]
    corrected_gallery_paths = [p for p, _ in included_gallery]
    corrected_gallery_labels = [lbl for _, lbl in included_gallery]

    # Validate counts against specification
    if len(query_paths) != config.EXPECTED_COUNTS["queries"]:
        raise ValueError(
            f"Expected {config.EXPECTED_COUNTS['queries']} queries, got {len(query_paths)}"
        )
    if len(corrected_gallery_paths) != config.EXPECTED_COUNTS["gallery"]:
        raise ValueError(
            f"Expected {config.EXPECTED_COUNTS['gallery']} gallery images, got {len(corrected_gallery_paths)}"
        )
    if len(excluded_paths) != config.EXPECTED_COUNTS["excluded_duplicate_gallery"]:
        raise ValueError(
            f"Expected {config.EXPECTED_COUNTS['excluded_duplicate_gallery']} excluded images, got {len(excluded_paths)}"
        )

    # Perform group-disjoint train/validation split on 899 gallery
    train_indices, val_indices = group_disjoint_split(
        corrected_gallery_paths,
        corrected_gallery_labels,
        group_ids,
        seed=mlp_seed
    )

    if len(train_indices) != config.EXPECTED_COUNTS["mlp_train"]:
        raise ValueError(
            f"Expected {config.EXPECTED_COUNTS['mlp_train']} train images, got {len(train_indices)}"
        )
    if len(val_indices) != config.EXPECTED_COUNTS["mlp_validation"]:
        raise ValueError(
            f"Expected {config.EXPECTED_COUNTS['mlp_validation']} val images, got {len(val_indices)}"
        )

    # Verify zero leakage
    train_paths = {corrected_gallery_paths[i] for i in train_indices}
    val_paths = {corrected_gallery_paths[i] for i in val_indices}

    if (train_paths | val_paths) & query_set:
        raise AssertionError("Leakage detected: query images present in training or validation set!")
    if (train_paths | val_paths) & excluded_paths:
        raise AssertionError("Leakage detected: excluded duplicate present in training or validation set!")

    return {
        "all_paths": all_paths,
        "query_paths": query_paths,
        "query_labels": query_labels,
        "gallery_paths": corrected_gallery_paths,
        "gallery_labels": corrected_gallery_labels,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_paths": [corrected_gallery_paths[i] for i in train_indices],
        "train_labels": [corrected_gallery_labels[i] for i in train_indices],
        "val_paths": [corrected_gallery_paths[i] for i in val_indices],
        "val_labels": [corrected_gallery_labels[i] for i in val_indices],
        "excluded_paths": sorted(list(excluded_paths)),
        "audit_results": audit_results,
        "group_ids": group_ids,
        "query_seed": query_seed,
        "mlp_seed": mlp_seed
    }


def save_split_manifest(
    split_info: Dict[str, Any],
    output_csv_path: Path | str,
    dataset_root: Path | None = None
) -> None:
    """
    Saves comprehensive split manifest to CSV format with exact provenance.
    """
    if dataset_root is None:
        dataset_root = config.DATASET_PATH

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    query_paths_set = set(split_info["query_paths"])
    train_paths_set = set(split_info["train_paths"])
    val_paths_set = set(split_info["val_paths"])
    excluded_paths_set = set(split_info["excluded_paths"])

    audit_results = split_info["audit_results"]
    sha256_map = audit_results["sha256_map"]
    phash_map = audit_results["phash_map"]
    group_ids = split_info["group_ids"]

    rows: List[Dict[str, Any]] = []

    for path in sorted(split_info["all_paths"]):
        rel_path = path.relative_to(dataset_root).as_posix()
        category = path.parent.name

        if path in query_paths_set:
            partition = "held_out_query"
            exclusion_reason = ""
        elif path in train_paths_set:
            partition = "mlp_train"
            exclusion_reason = ""
        elif path in val_paths_set:
            partition = "mlp_validation"
            exclusion_reason = ""
        elif path in excluded_paths_set:
            partition = "excluded_near_duplicate_of_query"
            exclusion_reason = "near/exact duplicate of held-out query"
        else:
            raise AssertionError(f"Path {rel_path} not assigned to any partition!")

        gid = group_ids.get(path, "")
        phash_hex = f"{phash_map[path]:016x}"

        rows.append({
            "relative_path": rel_path,
            "category": category,
            "partition": partition,
            "sha256": sha256_map[path],
            "perceptual_hash_hex": phash_hex,
            "duplicate_group_id": gid,
            "exclusion_reason": exclusion_reason,
            "query_split_seed": split_info["query_seed"],
            "mlp_split_seed": split_info["mlp_seed"]
        })

    fieldnames = [
        "relative_path",
        "category",
        "partition",
        "sha256",
        "perceptual_hash_hex",
        "duplicate_group_id",
        "exclusion_reason",
        "query_split_seed",
        "mlp_split_seed"
    ]

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
