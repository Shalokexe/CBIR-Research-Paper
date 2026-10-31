"""
Data handling, duplicate auditing, and deterministic partitioning package.
"""

from data.dataset_loader import load_dataset
from data.duplicate_audit import (
    compute_sha256,
    compute_perceptual_hash,
    audit_dataset_duplicates,
    build_group_id_map,
)
from data.partitioning import (
    create_corrected_splits,
    save_split_manifest,
)

__all__ = [
    "load_dataset",
    "compute_sha256",
    "compute_perceptual_hash",
    "audit_dataset_duplicates",
    "build_group_id_map",
    "create_corrected_splits",
    "save_split_manifest",
]
