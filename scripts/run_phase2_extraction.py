"""
Phase 2 Execution Script: Handcrafted Feature Extraction and Gallery Normalization.
Extracts 9-D HSV Color Moments and 4-D GLCM Texture Features for all images.
Fits Min-Max Normalizer on the 899 gallery images ONLY, preserving strict query isolation.
Generates:
- output/phase2/features.npz
- output/phase2/features.csv
- output/phase2/feature_metadata.json
- output/phase2/feature_extraction_config.json
- output/phase2/feature_validation_report.md
- figures/phase2_feature_distributions.png
- figures/phase2_representative_descriptors.png
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from data.dataset_loader import load_dataset
from data.partitioning import create_corrected_splits
from features.feature_extractor import (
    extract_hsv_color_features,
    extract_glcm_texture_features,
)
from features.normalization import MinMaxNormalizer


def run_phase2():
    print("=" * 70)
    print("ICAIN 2026 CBIR Implementation - Phase 2: Handcrafted Feature Extraction")
    print("=" * 70)

    dataset_root = config.DATASET_PATH
    output_dir = config.PHASE2_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load partitioned dataset from Phase 1 protocol
    print("\n[Step 1/5] Loading verified dataset partitions from Phase 1 protocol...")
    raw_gallery_paths, raw_gallery_labels, query_paths, query_labels = load_dataset(
        dataset_path=dataset_root, seed=config.RANDOM_SEED
    )
    split_info = create_corrected_splits(
        raw_gallery_paths=raw_gallery_paths,
        raw_gallery_labels=raw_gallery_labels,
        query_paths=query_paths,
        query_labels=query_labels,
        dataset_root=dataset_root,
        query_seed=config.RANDOM_SEED,
        mlp_seed=config.MLP_SEED
    )

    gallery_paths = split_info["gallery_paths"]  # 899 paths
    gallery_labels = split_info["gallery_labels"]
    query_paths = split_info["query_paths"]      # 100 paths
    query_labels = split_info["query_labels"]

    all_evaluated_paths = gallery_paths + query_paths  # 999 paths (899 gallery + 100 queries)
    all_evaluated_labels = gallery_labels + query_labels
    all_partitions = ["gallery"] * len(gallery_paths) + ["query"] * len(query_paths)
    rel_paths = [p.relative_to(dataset_root).as_posix() for p in all_evaluated_paths]

    print(f"  Gallery images: {len(gallery_paths)}")
    print(f"  Query images:   {len(query_paths)}")
    print(f"  Total evaluated images: {len(all_evaluated_paths)}")

    # 2. Extract HSV and GLCM features
    print("\n[Step 2/5] Extracting 9-D HSV color moments and 4-D GLCM texture features...")
    start_time = time.perf_counter()

    raw_color_list = []
    raw_texture_list = []
    extraction_times = []

    for idx, path in enumerate(all_evaluated_paths):
        t0 = time.perf_counter()
        c_feats = extract_hsv_color_features(path)
        t_feats = extract_glcm_texture_features(path)
        t1 = time.perf_counter()

        raw_color_list.append(c_feats)
        raw_texture_list.append(t_feats)
        extraction_times.append(t1 - t0)

        if (idx + 1) % 200 == 0 or (idx + 1) == len(all_evaluated_paths):
            print(f"  Processed {idx + 1}/{len(all_evaluated_paths)} images (mean: {np.mean(extraction_times)*1000:.2f} ms/image)...")

    total_extraction_time = time.perf_counter() - start_time
    print(f"  Extraction completed in {total_extraction_time:.2f} s ({total_extraction_time / len(all_evaluated_paths):.4f} s/image)")

    raw_color = np.array(raw_color_list, dtype=np.float32)      # (999, 9)
    raw_texture = np.array(raw_texture_list, dtype=np.float32)  # (999, 4)

    # Invariant validation
    assert raw_color.shape == (999, 9), f"Unexpected color shape: {raw_color.shape}"
    assert raw_texture.shape == (999, 4), f"Unexpected texture shape: {raw_texture.shape}"
    assert np.all(np.isfinite(raw_color)), "Non-finite values detected in color features!"
    assert np.all(np.isfinite(raw_texture)), "Non-finite values detected in texture features!"

    # 3. Fit Min-Max Normalizer on Gallery ONLY
    print("\n[Step 3/5] Fitting Min-Max Normalizer strictly on 899 gallery representations...")
    gallery_raw_color = raw_color[:len(gallery_paths)]      # Shape (899, 9)
    gallery_raw_texture = raw_texture[:len(gallery_paths)]  # Shape (899, 4)

    color_normalizer = MinMaxNormalizer()
    color_normalizer.fit(gallery_raw_color)

    texture_normalizer = MinMaxNormalizer()
    texture_normalizer.fit(gallery_raw_texture)

    # Transform all evaluated images using the gallery scaler
    normalized_color = color_normalizer.transform(raw_color)
    normalized_texture = texture_normalizer.transform(raw_texture)

    print("  Fitted Normalization Parameters:")
    print("  - Color Min:   ", np.round(color_normalizer.min_val, 4))
    print("  - Color Max:   ", np.round(color_normalizer.max_val, 4))
    print("  - Texture Min: ", np.round(texture_normalizer.min_val, 4))
    print("  - Texture Max: ", np.round(texture_normalizer.max_val, 4))

    # Gallery features must be strictly within [0.0, 1.0]
    assert np.all(normalized_color[:len(gallery_paths)] >= -1e-6)
    assert np.all(normalized_color[:len(gallery_paths)] <= 1.0 + 1e-6)
    assert np.all(normalized_texture[:len(gallery_paths)] >= -1e-6)
    assert np.all(normalized_texture[:len(gallery_paths)] <= 1.0 + 1e-6)

    # 4. Save data outputs and serialization
    print("\n[Step 4/5] Saving features, configuration, metadata, and validation report...")
    npz_path = output_dir / "features.npz"
    np.savez_compressed(
        npz_path,
        paths=np.array(rel_paths),
        partitions=np.array(all_partitions),
        labels=np.array(all_evaluated_labels),
        raw_color=raw_color,
        raw_texture=raw_texture,
        normalized_color=normalized_color,
        normalized_texture=normalized_texture,
        gallery_count=len(gallery_paths),
        query_count=len(query_paths)
    )
    print(f"  Saved compressed feature archive: {npz_path}")

    # Build comprehensive tabular CSV
    feature_columns = [
        "h_mean", "h_std", "h_skew",
        "s_mean", "s_std", "s_skew",
        "v_mean", "v_std", "v_skew",
        "contrast", "correlation", "energy", "homogeneity"
    ]
    raw_combined = np.hstack([raw_color, raw_texture])
    norm_combined = np.hstack([normalized_color, normalized_texture])

    csv_data = {
        "relative_path": rel_paths,
        "category": all_evaluated_labels,
        "partition": all_partitions
    }
    for i, col in enumerate(feature_columns):
        csv_data[f"raw_{col}"] = raw_combined[:, i]
    for i, col in enumerate(feature_columns):
        csv_data[f"norm_{col}"] = norm_combined[:, i]

    df_features = pd.DataFrame(csv_data)
    csv_path = output_dir / "features.csv"
    df_features.to_csv(csv_path, index=False)
    print(f"  Saved human-readable feature table: {csv_path}")

    # Save feature extraction configuration JSON
    config_dict = {
        "dataset_name": "Corel-1K",
        "dataset_path": str(dataset_root),
        "random_seed": config.RANDOM_SEED,
        "gallery_count": len(gallery_paths),
        "query_count": len(query_paths),
        "total_evaluated_images": len(all_evaluated_paths),
        "hsv_color_specification": {
            "dimensions": config.COLOR_FEATURE_DIM,
            "descriptors": ["mean", "std", "skewness"],
            "channels": ["H", "S", "V"],
            "conversion": "RGB -> HSV via matplotlib.colors.rgb_to_hsv",
            "value_ranges": {
                "H": "[0.0, 1.0)",
                "S": "[0.0, 1.0]",
                "V": "[0.0, 1.0]"
            },
            "calculation_method": "Pixel-based across flattened channel arrays",
            "skewness_formula": "mean((ch - mean)^3) / std^3",
            "zero_variance_threshold": 1e-6,
            "zero_variance_fallback": "skewness = 0.0"
        },
        "glcm_texture_specification": {
            "dimensions": config.TEXTURE_FEATURE_DIM,
            "properties": ["contrast", "correlation", "energy", "homogeneity"],
            "grayscale_conversion": "PIL 8-bit grayscale ('L' mode)",
            "quantization_levels": 256,
            "pixel_distance": config.GLCM_DISTANCE,
            "angles_degrees": config.GLCM_ANGLES_DEG,
            "angles_radians": [float(np.deg2rad(a)) for a in config.GLCM_ANGLES_DEG],
            "symmetric": True,
            "normed": True,
            "averaging_method": "Arithmetic mean of graycoprops across the four orientations"
        },
        "total_feature_dimensions": config.TOTAL_FEATURE_DIM,
        "normalization_protocol": {
            "algorithm": "Min-Max Normalization",
            "fitting_scope": "Gallery images ONLY (899 images)",
            "query_isolation": "Held-out queries strictly excluded from parameter fitting",
            "zero_range_threshold": 1e-8,
            "zero_range_fallback": "range = 1.0"
        }
    }
    config_json_path = output_dir / "feature_extraction_config.json"
    with config_json_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
    print(f"  Saved configuration: {config_json_path}")

    # Save feature metadata JSON
    metadata_dict = {
        "extraction_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_images_processed": len(all_evaluated_paths),
        "total_extraction_seconds": float(total_extraction_time),
        "mean_extraction_seconds_per_image": float(total_extraction_time / len(all_evaluated_paths)),
        "color_dimensions": 9,
        "texture_dimensions": 4,
        "total_dimensions": 13,
        "normalization_parameters": {
            "color_min": [float(v) for v in color_normalizer.min_val],
            "color_max": [float(v) for v in color_normalizer.max_val],
            "texture_min": [float(v) for v in texture_normalizer.min_val],
            "texture_max": [float(v) for v in texture_normalizer.max_val]
        },
        "feature_statistics_gallery": {
            "raw_color_mean": [float(v) for v in np.mean(gallery_raw_color, axis=0)],
            "raw_color_std": [float(v) for v in np.std(gallery_raw_color, axis=0)],
            "raw_texture_mean": [float(v) for v in np.mean(gallery_raw_texture, axis=0)],
            "raw_texture_std": [float(v) for v in np.std(gallery_raw_texture, axis=0)]
        }
    }
    metadata_json_path = output_dir / "feature_metadata.json"
    with metadata_json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"  Saved metadata: {metadata_json_path}")

    # 5. Generate Visualizations
    print("\n[Step 5/5] Generating publication-quality feature visualizations...")

    # Visualization 1: Grouped Boxplot of Normalized Descriptors by Modality
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    color_labels = ["H μ", "H σ", "H s", "S μ", "S σ", "S s", "V μ", "V σ", "V s"]
    axes[0].boxplot(normalized_color[:len(gallery_paths)], patch_artist=True,
                    boxprops=dict(facecolor="#2b5c8f", color="#1c3b5e", alpha=0.8),
                    medianprops=dict(color="#d95f02", linewidth=1.5))
    axes[0].set_title("Normalized HSV Color Moments (Gallery: N=899)", fontsize=12, fontweight="bold")
    axes[0].set_xticklabels(color_labels, fontsize=10)
    axes[0].set_ylabel("Normalized Value [0, 1]", fontsize=11)
    axes[0].set_ylim(-0.05, 1.05)

    texture_labels = ["Contrast", "Correlation", "Energy", "Homogeneity"]
    axes[1].boxplot(normalized_texture[:len(gallery_paths)], patch_artist=True,
                    boxprops=dict(facecolor="#2e7d32", color="#1b5e20", alpha=0.8),
                    medianprops=dict(color="#d95f02", linewidth=1.5))
    axes[1].set_title("Normalized GLCM Texture Features (Gallery: N=899)", fontsize=12, fontweight="bold")
    axes[1].set_xticklabels(texture_labels, fontsize=10)
    axes[1].set_ylabel("Normalized Value [0, 1]", fontsize=11)
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    dist_plot_path = figures_dir / "phase2_feature_distributions.png"
    fig.savefig(dist_plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature distribution plot: {dist_plot_path}")

    # Visualization 2: Category Profiles (Color vs Texture Energy/Contrast)
    cat_means_color = {}
    cat_means_texture = {}
    for cat in config.EXPECTED_CATEGORIES:
        mask = [lbl == cat for lbl in gallery_labels]
        cat_means_color[cat] = np.mean(normalized_color[:len(gallery_paths)][mask], axis=0)
        cat_means_texture[cat] = np.mean(normalized_texture[:len(gallery_paths)][mask], axis=0)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    categories = config.EXPECTED_CATEGORIES
    x = np.arange(len(categories))
    width = 0.35

    # Mean Color Intensity vs Mean Texture Contrast across categories
    mean_color_var = [cat_means_color[c][3] for c in categories]   # S mean
    mean_texture_con = [cat_means_texture[c][0] for c in categories] # Contrast

    rects1 = ax.bar(x - width/2, mean_color_var, width, label="Mean Saturation (S μ)", color="#1f77b4", alpha=0.85)
    rects2 = ax.bar(x + width/2, mean_texture_con, width, label="Mean Texture Contrast", color="#2ca02c", alpha=0.85)

    ax.set_title("Category-Wise Visual Signatures: Color Saturation vs. Texture Contrast", fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalized Mean Value", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9, rotation=0)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    cat_plot_path = figures_dir / "phase2_representative_descriptors.png"
    fig.savefig(cat_plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved representative descriptor plot: {cat_plot_path}")

    # 6. Generate feature_validation_report.md
    validation_report_content = f"""# Phase 2 Feature Extraction and Normalization Validation Report

## 1. Feature Representation Summary
- **Total Images Processed**: {len(all_evaluated_paths)} (899 Gallery + 100 Held-Out Queries)
- **Feature Vector Dimensionality**: Exactly 13 dimensions
  - **HSV Color Features**: 9 dimensions (Mean, Standard Deviation, Skewness for H, S, V)
  - **GLCM Texture Features**: 4 dimensions (Contrast, Correlation, Energy, Homogeneity averaged over 4 orientations)
- **Feature Integrity**:
  - Missing values: 0
  - Non-finite values (NaN / Inf): 0
  - Constant channel zero-variance handling: Fully protected ($\sigma < 10^{{-6}} \implies s = 0.0$)

## 2. Methodological Specifications
### HSV Color Feature Extraction
- **Color Conversion**: PIL standard RGB cast to `float32` and scaled to $[0.0, 1.0]$ via division by $255.0$. Converted to HSV via `matplotlib.colors.rgb_to_hsv`.
- **Value Ranges**: $H \in [0.0, 1.0)$, $S \in [0.0, 1.0]$, $V \in [0.0, 1.0]$.
- **Calculation Type**: Pixel-based calculation across flattened channel arrays.
- **Moments**:
  $$\mu = \\frac{{1}}{{N}} \\sum x_i, \\quad \\sigma = \\sqrt{{\\frac{{1}}{{N}} \\sum (x_i - \\mu)^2}}, \\quad s = \\frac{{\\frac{{1}}{{N}} \\sum (x_i - \\mu)^3}}{{\\sigma^3}}$$
- **Zero Variance Fallback**: When $\sigma < 10^{{-6}}$, skewness is assigned exactly $0.0$.

### GLCM Texture Feature Extraction
- **Grayscale Conversion**: PIL 8-bit grayscale mode (`'L'`) with values in $[0, 255]$.
- **Quantization Levels**: 256 gray levels (`levels=256`).
- **Pixel Distance**: 1 pixel (`distances=[1]`).
- **Orientations**: 4 directions: $0^\\circ, 45^\\circ, 90^\\circ, 135^\\circ$ ($0, \\pi/4, \\pi/2, 3\\pi/4$ radians).
- **Matrix Settings**: `symmetric=True`, `normed=True`.
- **Directional Averaging**: Computed via scikit-image's `graycoprops` and averaged:
  $$\\text{{Property}} = \\frac{{1}}{{4}} \\sum_{{\\theta \\in \\{{0^\\circ, 45^\\circ, 90^\\circ, 135^\\circ\\}}}} \\text{{graycoprops}}(GLCM_\\theta, \\text{{Property}})$$

## 3. Gallery-Only Min-Max Normalization
- **Strict Leakage Prevention**: Parameters $\\min$ and $\\max$ were fitted exclusively on the 899 gallery images.
- **Fitted Bounds**:
  - **Color Min**: `{np.array2string(color_normalizer.min_val, precision=4, separator=', ')}`
  - **Color Max**: `{np.array2string(color_normalizer.max_val, precision=4, separator=', ')}`
  - **Texture Min**: `{np.array2string(texture_normalizer.min_val, precision=4, separator=', ')}`
  - **Texture Max**: `{np.array2string(texture_normalizer.max_val, precision=4, separator=', ')}`
- **Gallery Output Range**: All 899 gallery normalized values lie strictly within $[0.0, 1.0]$.
- **Query Set Isolation**: Held-out queries were transformed using the frozen gallery scaler without updating parameter state.

## 4. Extraction Timing and Performance
- **Total Extraction Duration**: {total_extraction_time:.2f} seconds
- **Mean Processing Time per Image**: {total_extraction_time / len(all_evaluated_paths):.4f} seconds ({total_extraction_time / len(all_evaluated_paths) * 1000:.2f} ms/image)

## 5. Artifacts Generated
- `output/phase2/features.npz`: Compressed archive with arrays `raw_color`, `raw_texture`, `normalized_color`, `normalized_texture`.
- `output/phase2/features.csv`: Tabular CSV containing raw and normalized features for all 999 evaluated images.
- `output/phase2/feature_extraction_config.json`: Complete specification file enabling deterministic reproduction.
- `output/phase2/feature_metadata.json`: Serialized extraction metrics, timestamps, and normalization bounds.
- `figures/phase2_feature_distributions.png`: High-resolution boxplot of color and texture distributions.
- `figures/phase2_representative_descriptors.png`: Category-level descriptor signature comparisons.

## 6. Verification Status
- **Validation**: **PASSED**.
- All mathematical bounds, shapes, and zero-leakage constraints verified successfully.
"""
    val_report_path = output_dir / "feature_validation_report.md"
    val_report_path.write_text(validation_report_content, encoding="utf-8")
    print(f"  Saved validation report: {val_report_path}")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE: Feature extraction and normalization successfully verified.")
    print("=" * 70)


if __name__ == "__main__":
    run_phase2()
