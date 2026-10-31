"""
Corel-1K Dataset Loader Module.
Discovers and validates all 1,000 images across 10 categories,
and executes deterministic query/gallery splitting.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import config


def load_dataset(
    dataset_path: Path | str | None = None,
    seed: int | None = None
) -> Tuple[List[Path], List[str], List[Path], List[str]]:
    """
    Recursively scans the dataset directory, validates category structures,
    and splits images deterministically into initial gallery and query sets.

    Args:
        dataset_path: Path to dataset root directory. Defaults to config.DATASET_PATH.
        seed: Random seed for deterministic query selection. Defaults to config.RANDOM_SEED.

    Returns:
        gallery_paths: List of absolute Path objects for gallery images.
        gallery_labels: Category names for each gallery image.
        query_paths: List of absolute Path objects for query images.
        query_labels: Category names for each query image.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
        ValueError: If categories do not contain exactly 100 images.
    """
    if dataset_path is None:
        dataset_path = config.DATASET_PATH
    if seed is None:
        seed = config.RANDOM_SEED

    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # 1. Discover all image files recursively (case-insensitive extension matching)
    image_paths: List[Path] = []
    for ext in config.IMAGE_EXTENSIONS:
        image_paths.extend(dataset_path.rglob(f"*{ext}"))
        image_paths.extend(dataset_path.rglob(f"*{ext.upper()}"))

    # Deduplicate and sort for absolute determinism
    unique_paths = sorted(list(set(p.resolve() for p in image_paths)))
    if not unique_paths:
        raise FileNotFoundError(f"No valid image files found under {dataset_path}")

    # 2. Group paths by their immediate parent folder (semantic category)
    expected_cats = set(config.EXPECTED_CATEGORIES)
    category_to_paths: Dict[str, List[Path]] = {cat: [] for cat in config.EXPECTED_CATEGORIES}

    for p in unique_paths:
        cat_name = p.parent.name
        if cat_name in expected_cats:
            category_to_paths[cat_name].append(p)

    # 3. Validate image counts per category
    for cat in config.EXPECTED_CATEGORIES:
        count = len(category_to_paths[cat])
        if count != config.TOTAL_IMAGES_PER_CAT:
            raise ValueError(
                f"Category '{cat}' must contain exactly {config.TOTAL_IMAGES_PER_CAT} images; found {count}."
            )

    total_images = sum(len(paths) for paths in category_to_paths.values())
    if total_images != config.EXPECTED_COUNTS["dataset_images"]:
        raise ValueError(
            f"Expected {config.EXPECTED_COUNTS['dataset_images']} images; found {total_images}."
        )

    # 4. Deterministic query-gallery selection using fixed random generator
    rng = np.random.default_rng(seed=seed)
    gallery_paths: List[Path] = []
    gallery_labels: List[str] = []
    query_paths: List[Path] = []
    query_labels: List[str] = []

    for cat in sorted(config.EXPECTED_CATEGORIES):
        paths = sorted(category_to_paths[cat])
        indices = np.arange(len(paths))
        rng.shuffle(indices)

        # Select first 10 shuffled indices as queries, remaining 90 as gallery
        query_indices = set(indices[:config.QUERY_IMAGES_PER_CAT])

        for idx, path in enumerate(paths):
            if idx in query_indices:
                query_paths.append(path)
                query_labels.append(cat)
            else:
                gallery_paths.append(path)
                gallery_labels.append(cat)

    return gallery_paths, gallery_labels, query_paths, query_labels
