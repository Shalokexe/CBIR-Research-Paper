"""
Duplicate Auditing and Perceptual Hashing Module for Corel-1K.
Provides cryptographic (SHA-256) and perceptual (32x32 DCT) hashing to identify
exact duplicates and near-duplicate image pairs.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import numpy as np
from PIL import Image
from scipy.fft import dctn

import config


def compute_sha256(path: Path | str) -> str:
    """
    Computes cryptographic SHA-256 digest of a file.

    Args:
        path: Path to file.

    Returns:
        Hexadecimal SHA-256 string.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_perceptual_hash(path: Path | str) -> int:
    """
    Computes a 64-bit DCT-based perceptual hash (pHash) for an image.
    Protocol: 32x32 grayscale Lanczos resize, 2D DCT type-2 orthonormal,
    top-left 8x8 low-frequency coefficients, median threshold (excluding DC term).

    Args:
        path: Path to image file.

    Returns:
        64-bit integer representing the perceptual hash.
    """
    with Image.open(path) as img:
        gray = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)

    # 2D Discrete Cosine Transform
    coefficients = dctn(pixels, type=2, norm="ortho")[:8, :8].reshape(-1)

    # Median threshold computed excluding the DC coefficient (coefficients[0])
    median = float(np.median(coefficients[1:]))
    bits = coefficients > median

    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def audit_dataset_duplicates(
    image_paths: List[Path],
    hamming_threshold: int = config.NEAR_DUPLICATE_HAMMING_THRESHOLD
) -> Dict[str, Any]:
    """
    Performs full SHA-256 and pHash inventory across all provided image paths.
    Identifies exact duplicate and near-duplicate pairs under the specified Hamming distance threshold.

    Args:
        image_paths: List of absolute image paths.
        hamming_threshold: Maximum Hamming distance to consider as a near duplicate.

    Returns:
        Dictionary containing:
            - 'all_paths': sorted unique paths
            - 'sha256_map': dict mapping path -> sha256
            - 'phash_map': dict mapping path -> int phash
            - 'exact_pairs': list of dicts with exact duplicate pairs
            - 'near_pairs': list of dicts with near-duplicate pairs and distances
    """
    unique_paths = sorted(list(set(image_paths)))
    sha256_map: Dict[Path, str] = {}
    phash_map: Dict[Path, int] = {}
    exact_groups: Dict[str, List[Path]] = {}

    for path in unique_paths:
        digest = compute_sha256(path)
        sha256_map[path] = digest
        exact_groups.setdefault(digest, []).append(path)
        phash_map[path] = compute_perceptual_hash(path)

    # Exact duplicates (identical SHA-256)
    exact_pairs: List[Dict[str, Any]] = []
    for digest, paths in exact_groups.items():
        if len(paths) > 1:
            for i, p1 in enumerate(paths):
                for p2 in paths[i + 1:]:
                    exact_pairs.append({
                        "left_path": p1,
                        "right_path": p2,
                        "sha256": digest
                    })

    # Near duplicates (Hamming distance on pHash <= threshold, excluding exact duplicates)
    near_pairs: List[Dict[str, Any]] = []
    n = len(unique_paths)
    for i in range(n):
        p1 = unique_paths[i]
        h1 = phash_map[p1]
        for j in range(i + 1, n):
            p2 = unique_paths[j]
            h2 = phash_map[p2]
            dist = (h1 ^ h2).bit_count()
            if dist <= hamming_threshold:
                near_pairs.append({
                    "left_path": p1,
                    "right_path": p2,
                    "distance": dist
                })

    return {
        "all_paths": unique_paths,
        "sha256_map": sha256_map,
        "phash_map": phash_map,
        "exact_pairs": exact_pairs,
        "near_pairs": near_pairs
    }


def build_group_id_map(
    exact_pairs: List[Dict[str, Any]],
    near_pairs: List[Dict[str, Any]]
) -> Dict[Path, str]:
    """
    Constructs a connected-component duplicate group ID mapping using Disjoint-Set / Union-Find.

    Args:
        exact_pairs: List of exact duplicate pair dicts.
        near_pairs: List of near-duplicate pair dicts.

    Returns:
        Mapping from Path to group ID string (e.g. 'duplicate-0001').
    """
    parents: Dict[Path, Path] = {}

    def find(p: Path) -> Path:
        parents.setdefault(p, p)
        if parents[p] != p:
            parents[p] = find(parents[p])
        return parents[p]

    def union(p1: Path, p2: Path) -> None:
        root1 = find(p1)
        root2 = find(p2)
        if root1 != root2:
            # Deterministic ordering
            if str(root1) < str(root2):
                parents[root2] = root1
            else:
                parents[root1] = root2

    for pair in exact_pairs:
        union(pair["left_path"], pair["right_path"])
    for pair in near_pairs:
        union(pair["left_path"], pair["right_path"])

    groups: Dict[Path, List[Path]] = {}
    for p in list(parents):
        root = find(p)
        groups.setdefault(root, []).append(p)

    ordered = sorted([sorted(members) for members in groups.values()], key=lambda m: str(m[0]))
    id_map: Dict[Path, str] = {}
    for num, members in enumerate(ordered, start=1):
        group_id = f"duplicate-{num:04d}"
        for member in members:
            id_map[member] = group_id

    return id_map
