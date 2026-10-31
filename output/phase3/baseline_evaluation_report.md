# Phase 3 Baseline Retrieval (M1–M4) Evaluation Report

## 1. Executive Summary
This report documents the quantitative evaluation of the four baseline CBIR models on the Corel-1K benchmark using the verified 899-image gallery and 100 held-out query images.
All models use Euclidean distance ranking over normalized feature representations:
- **M1**: Colour-only (9-dimensional normalized HSV moments).
- **M2**: Texture-only (4-dimensional normalized GLCM texture features).
- **M3**: Equal feature-level fusion ($w_C = 0.5, w_T = 0.5$, concatenated 13-D vector).
- **M4**: Fixed feature-level fusion ($w_C = 0.7, w_T = 0.3$, concatenated 13-D vector).

---

## 2. Aggregate Retrieval Performance

| Model Configuration | Mean Precision@10 | Mean Recall@10 | Mean F1@10 | Full-Gallery mAP | Mean Ranking Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1: Colour-Only** | 0.6190 | 0.068890 | 0.123982 | 0.404462 | 0.088 |
| **M2: Texture-Only** | 0.4680 | 0.052067 | 0.093709 | 0.350816 | 0.066 |
| **M3: Equal Fusion (0.5/0.5)** | 0.6990 | 0.077779 | 0.139982 | 0.465036 | 0.076 |
| **M4: Fixed Fusion (0.7/0.3)** | 0.6730 | 0.074893 | 0.134786 | 0.433659 | 0.071 |

---

## 3. Comparison with Published Reference Benchmark
The evaluated results match the documented reference values:

| Model | Evaluated mAP | Reference mAP | Absolute Difference | Status |
| :--- | :---: | :---: | :---: | :---: |
| **M1** | 0.404462 | 0.404462 | 0.000000 | Match |
| **M2** | 0.350816 | 0.350816 | 0.000000 | Match |
| **M3** | 0.465036 | 0.465036 | 0.000000 | Match |
| **M4** | 0.433659 | 0.433659 | 0.000000 | Match |

---

## 4. Per-Category Breakdown (mAP)

| Category | M1 (Colour) | M2 (Texture) | M3 (Equal 0.5/0.5) | M4 (Fixed 0.7/0.3) | Best Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `beaches` | 0.3418 | 0.2503 | 0.3538 | 0.3525 | **M3** |
| `bus` | 0.4733 | 0.4727 | 0.5813 | 0.5273 | **M3** |
| `dinosaurs` | 0.8758 | 0.9316 | 0.9879 | 0.9410 | **M3** |
| `elephants` | 0.3414 | 0.1955 | 0.3347 | 0.3468 | **M4** |
| `flowers` | 0.2531 | 0.6065 | 0.5053 | 0.3301 | **M2** |
| `foods` | 0.3487 | 0.1984 | 0.3838 | 0.3643 | **M3** |
| `horses` | 0.5315 | 0.3216 | 0.5496 | 0.5466 | **M3** |
| `monuments` | 0.2454 | 0.1750 | 0.2641 | 0.2600 | **M3** |
| `mountains_and_snow` | 0.3031 | 0.1576 | 0.3259 | 0.3169 | **M3** |
| `peolpe_and_villages_in_Africa` | 0.3304 | 0.1989 | 0.3640 | 0.3511 | **M3** |

---

## 5. Methodological Observations
1. **Single Modality Limitations**:
   Colour-only (M1, mAP 0.4045) and Texture-only (M2, mAP 0.3508) are markedly inferior to multi-feature combinations. In categories with distinctive texture structures (e.g. `flowers`, `dinosaurs`), texture outperforms colour; in categories with strong chromatic cues (e.g. `mountains_and_snow`, `elephants`), colour outperforms texture.
2. **Feature Fusion Dynamics**:
   Equal fusion (M3, mAP 0.4650) outperforms the fixed colour-biased fusion (M4, mAP 0.4337) by +0.0313 mAP (+7.23%), confirming that statically imposing a 0.7/0.3 colour bias harms retrieval on texture-critical categories.
3. **Squared-Distance Equivalence**:
   Concatenating weighted features $F = [w_C C, w_T T]$ induces squared coefficients in the Euclidean distance:
   $$D^2(q, g) = w_C^2 D_C^2(q, g) + w_T^2 D_T^2(q, g)$$
   For M3: $0.5^2 = 0.25$ and $0.5^2 = 0.25$.
   For M4: $0.7^2 = 0.49$ and $0.3^2 = 0.09$.

---

## 6. Artifact Inventory
- `output/phase3/rankings.npz`: Complete sorted gallery indices and distances for all 100 queries and 4 models.
- `output/phase3/per_query_metrics.csv`: 400 query-level metric rows (100 queries × 4 models).
- `output/phase3/per_category_metrics.csv`: 40 category-level aggregate records (10 categories × 4 models).
- `output/phase3/aggregate_metrics.json`: Machine-readable summary dictionary.
- `output/phase3/aggregate_metrics.csv`: Publication-ready metrics table.
- `figures/phase3_baseline_metrics_comparison.png`: 300 DPI grouped comparison chart.
