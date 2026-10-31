# Phase 4 CBIR Retrieval Engine & Full M1–M6 Evaluation Report

## 1. Executive Summary
This report presents the complete quantitative evaluation of all six content-based image retrieval models (M1–M6) on the Corel-1K dataset using the corrected, duplicate-audited protocol (899 gallery images, 100 held-out query images).
The proposed AI-weighted model (**M5**) achieves the highest retrieval accuracy across all metrics, yielding **0.7100 Precision@10** and **0.4798 mAP**.

---

## 2. Overall Performance Comparison (M1–M6)

| Model Configuration | Mean Precision@10 | Mean Recall@10 | Mean F1@10 | Full-Gallery mAP | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1: Colour-Only** | 0.6190 | 0.068890 | 0.123982 | 0.404462 | 0.102 |
| **M2: Texture-Only** | 0.4680 | 0.052067 | 0.093709 | 0.350816 | 0.080 |
| **M3: Equal Fusion (0.5/0.5)** | 0.6990 | 0.077779 | 0.139982 | 0.465036 | 0.101 |
| **M4: Fixed Fusion (0.7/0.3)** | 0.6730 | 0.074893 | 0.134786 | 0.433659 | 0.094 |
| **M5: Proposed Adaptive** | **0.7100** | **0.079000** | **0.142180** | **0.479794** | 0.246 |
| **M6: Fixed Distance Control** | 0.6960 | 0.077443 | 0.139378 | 0.460536 | 0.246 |

---

## 3. Comparison with Published Reference Benchmark

| Model | Evaluated mAP | Reference Benchmark mAP | Absolute Difference | Parity Status |
| :--- | :---: | :---: | :---: | :---: |
| **M1** | 0.404462 | 0.404462 | 0.000000 | **Exact Match** |
| **M2** | 0.350816 | 0.350816 | 0.000000 | **Exact Match** |
| **M3** | 0.465036 | 0.465036 | 0.000000 | **Exact Match** |
| **M4** | 0.433659 | 0.433659 | 0.000000 | **Exact Match** |
| **M5** | 0.479794 | 0.479794 | 0.000000 | **Exact Match** |
| **M6** | 0.460536 | 0.460536 | 0.000000 | **Exact Match** |

* Exact match across all six models: 0 mismatches in rankings, metric outputs, and weight means.

---

## 4. Per-Category Breakdown (mAP)

| Category | M1 | M2 | M3 | M4 | M5 | M6 | Best Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `beaches` | 0.3418 | 0.2503 | 0.3538 | 0.3525 | 0.3563 | 0.3558 | **M5** |
| `bus` | 0.4733 | 0.4727 | 0.5813 | 0.5273 | 0.5567 | 0.5732 | **M3** |
| `dinosaurs` | 0.8758 | 0.9316 | 0.9879 | 0.9410 | 0.9886 | 0.9831 | **M5** |
| `elephants` | 0.3414 | 0.1955 | 0.3347 | 0.3468 | 0.3444 | 0.3392 | **M4** |
| `flowers` | 0.2531 | 0.6065 | 0.5053 | 0.3301 | 0.6726 | 0.4623 | **M5** |
| `foods` | 0.3487 | 0.1984 | 0.3838 | 0.3643 | 0.3770 | 0.3817 | **M3** |
| `horses` | 0.5315 | 0.3216 | 0.5496 | 0.5466 | 0.5519 | 0.5525 | **M6** |
| `monuments` | 0.2454 | 0.1750 | 0.2641 | 0.2600 | 0.2656 | 0.2654 | **M5** |
| `mountains_and_snow` | 0.3031 | 0.1576 | 0.3259 | 0.3169 | 0.3206 | 0.3273 | **M6** |
| `peolpe_and_villages_in_Africa` | 0.3304 | 0.1989 | 0.3640 | 0.3511 | 0.3642 | 0.3649 | **M6** |

---

## 5. Statistical Dispersion of MLP-Predicted Weights (M5)

| Modality Component | Mean | Standard Deviation (SD) | Minimum | Maximum |
| :--- | :---: | :---: | :---: | :---: |
| **Colour Weight ($w_C$)** | 0.5784 | 0.1568 | 0.0956 | 0.7523 |
| **Texture Weight ($w_T$)** | 0.4216 | 0.1568 | 0.2477 | 0.9044 |

The substantial standard deviation of the predicted weights demonstrates that the model actively adapts the weights to each query's visual characteristics rather than collapsing onto a static combination.

---

## 6. Artifact Inventory
- `output/phase4/rankings.npz`: Sorted gallery indices and distances for all 6 models and 100 queries.
- `output/phase4/query_weights.csv`: 100 query weight predictions for M5 and M6.
- `output/phase4/per_query_metrics.csv`: 600 query-level metric rows (100 queries × 6 models).
- `output/phase4/per_category_metrics.csv`: 60 category-level aggregate records (10 categories × 6 models).
- `output/phase4/aggregate_metrics.json`: Machine-readable summary dictionary.
- `output/phase4/aggregate_metrics.csv`: Summary metrics table.
- `output/phase4/retrieval_config.json`: Complete specification of parameters and models.
- `figures/phase4_m1_m6_performance_comparison.png`: 300 DPI grouped bar chart for M1–M6.
- `figures/phase4_m5_weight_distribution.png`: Histogram of predicted weights.
- `figures/phase4_category_weights.png`: Category-wise average predicted weights.
