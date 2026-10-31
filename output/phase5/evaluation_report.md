# Phase 5 — Corrected Corel-1K Benchmark Evaluation Report

**Conference:** ICAIN 2026, Springer LNNS — Paper ID 1567  
**Title:** Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features  
**Evaluation Scope:** Full M1–M6 Benchmark, Per-Query Metrics, Paired Inferential Statistics, Category Sensitivity, and Flowers Outlier Analysis  
**Date:** 2026-10-31 23:44:11  

---

## 1. Experimental Protocol & Dataset Specifications

All evaluations in Phase 5 strictly follow the corrected, leakage-free Corel-1K evaluation protocol:
- **Total Dataset Images:** 1,000 images across 10 semantic categories (100 images per category).
- **Near-Duplicate Audit Exclusion:** `training_set/horses/761.jpg` was permanently excluded from the gallery due to perceptual hash collision (Hamming distance 4 with query counterpart `776.jpg`).
- **Corrected Gallery:** **899 images** (700 internal training images + 199 validation images).
- **Held-Out Query Set:** **100 images** (10 images per category), strictly isolated from model training, validation, early stopping, hyperparameter tuning, and MinMax normalization parameter fitting.
- **Feature Space:** 13 handcrafted dimensions (9 HSV colour moments + 4 GLCM texture properties averaged over four angles: 0°, 45°, 90°, 135°).
- **Cut-off Rank ($K$):** 10.
- **Aggregation Convention:** Per-query metrics calculated individually, unrounded per-query $F1@10 = \frac{2 \cdot P@10 \cdot R@10}{P@10 + R@10}$, and aggregated as arithmetic means across queries.

---

## 2. Overall Performance Comparison (M1–M6)

| Model Architecture | Fusion Paradigm | Distance Formulation | Precision@10 | Recall@10 | F1@10 | mAP | Mean Latency (ms) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **M1: Colour-Only** | Single Modality | $D_C$ (9D HSV) | 0.6190 | 0.068890 | 0.123982 | 0.404462 | 0.202 |
| **M2: Texture-Only** | Single Modality | $D_T$ (4D GLCM) | 0.4680 | 0.052067 | 0.093709 | 0.350816 | 0.115 |
| **M3: Equal Fusion** | Feature-Level | Concatenation $[0.5C, 0.5T]$ | 0.6990 | 0.077779 | 0.139982 | 0.465036 | 0.408 |
| **M4: Fixed Fusion** | Feature-Level | Concatenation $[0.7C, 0.3T]$ | 0.6730 | 0.074893 | 0.134786 | 0.433659 | 0.134 |
| **M5: Proposed Adaptive** | Distance-Level | Adaptive $w_C(q) D_C^2 + w_T(q) D_T^2$ | **0.7100** | **0.079000** | **0.142180** | **0.479794** | 0.365 |
| **M6: Fixed Distance Control** | Distance-Level | Fixed $\bar{w}_C D_C^2 + \bar{w}_T D_T^2$ | 0.6960 | 0.077443 | 0.139378 | 0.460536 | 0.343 |

---

## 3. Reference Parity Audit Target Verification

| Model | Evaluated mAP | Reference Benchmark mAP | Absolute Difference | Verification Status |
|:---|:---:|:---:|:---:|:---:|
| **M1** | 0.404462 | 0.404462 | 0.000000 | **Exact Match** |
| **M2** | 0.350816 | 0.350816 | 0.000000 | **Exact Match** |
| **M3** | 0.465036 | 0.465036 | 0.000000 | **Exact Match** |
| **M4** | 0.433659 | 0.433659 | 0.000000 | **Exact Match** |
| **M5** | 0.479794 | 0.479794 | 0.000000 | **Exact Match** |
| **M6** | 0.460536 | 0.460536 | 0.000000 | **Exact Match** |

* All 6 models match reference audit targets to 6 decimal places ($0.000000$ absolute error).

---

## 4. Query-Level Paired Statistical Analysis (M5 vs M3 and M5 vs M6)

To rigorously evaluate whether M5's observed aggregate retrieval gains constitute statistically significant improvements, non-parametric two-sided Wilcoxon signed-rank tests and stratified paired bootstrap confidence intervals were executed across all 100 queries:

| Comparison | Mean $\Delta AP$ | Sample SD | Wins | Ties | Losses | Wilcoxon $W$ | Raw $p$-value | Holm-Adjusted $p$ | Significant ($lpha=0.05$)? | Stratified Bootstrap 95% CI |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M5 vs M3** | **+0.014758** | 0.065948 | 51 | 6 | 43 | 1972.0 | 0.3259 | **0.6519** | **NO** | [0.007414, 0.023385] |
| **M5 vs M6** | **+0.019258** | 0.076450 | 53 | 0 | 47 | 2353.0 | 0.5543 | **0.6519** | **NO** | [0.011935, 0.027971] |

### Critical Inferential Insights:
1. **Lack of Significance:** Neither paired comparison reaches statistical significance ($p = 0.3259$ for M5 vs M3 and $p = 0.5543$ for M5 vs M6; Holm-adjusted $p = 0.6519$ for both).
2. **Win/Loss Balance:** Across individual queries, M5 wins 51 vs 43 losses against M3, and 53 wins vs 47 losses against M6. The query-level win rates are nearly balanced ($pprox 50\%$), confirming why rank-based sign tests do not reject the null hypothesis.
3. **Bootstrap CI vs Wilcoxon Distinction:** The positive bootstrap confidence interval reflects the arithmetic mean difference driven by large asymmetric gains in specific queries (notably *flowers*), whereas the Wilcoxon signed-rank test evaluates the distribution of ranks and signs. A positive confidence interval does **not** render the hypothesis test significant.

---

## 5. Category-Level Sensitivity & The Flowers Outlier Effect

Per-category Average Precision breakdown across all six configurations:

| Category | M1 (Colour) | M2 (Texture) | M3 (Equal) | M4 (0.7/0.3) | M5 (Adaptive) | M6 (Fixed Control) | Best Model | $\Delta(M5-M3)$ | $\Delta(M5-M6)$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `beaches` | 0.3418 | 0.2503 | 0.3538 | 0.3525 | 0.3563 | 0.3558 | **M5** | +0.0025 | +0.0005 |
| `bus` | 0.4733 | 0.4727 | 0.5813 | 0.5273 | 0.5567 | 0.5732 | **M3** | -0.0246 | -0.0165 |
| `dinosaurs` | 0.8758 | 0.9316 | 0.9879 | 0.9410 | 0.9886 | 0.9831 | **M5** | +0.0007 | +0.0055 |
| `elephants` | 0.3414 | 0.1955 | 0.3347 | 0.3468 | 0.3444 | 0.3392 | **M4** | +0.0097 | +0.0052 |
| `flowers` | 0.2531 | 0.6065 | 0.5053 | 0.3301 | 0.6726 | 0.4623 | **M5** | +0.1673 | +0.2103 |
| `foods` | 0.3487 | 0.1984 | 0.3838 | 0.3643 | 0.3770 | 0.3817 | **M3** | -0.0068 | -0.0047 |
| `horses` | 0.5315 | 0.3216 | 0.5496 | 0.5466 | 0.5519 | 0.5525 | **M6** | +0.0024 | -0.0005 |
| `monuments` | 0.2454 | 0.1750 | 0.2641 | 0.2600 | 0.2656 | 0.2654 | **M5** | +0.0015 | +0.0002 |
| `mountains_and_snow` | 0.3031 | 0.1576 | 0.3259 | 0.3169 | 0.3206 | 0.3273 | **M6** | -0.0053 | -0.0066 |
| `peolpe_and_villages_in_Africa` | 0.3304 | 0.1989 | 0.3640 | 0.3511 | 0.3642 | 0.3649 | **M6** | +0.0002 | -0.0008 |

### Quantification of the Flowers Category Driver Effect:

- **Flowers Category Alone:**
  - $\Delta AP(M5 - M3) = \mathbf{+0.167349}$
  - $\Delta AP(M5 - M6) = \mathbf{+0.210336}$
- **All 9 Remaining Categories Combined (Excluding Flowers):**
  - $\Delta AP(M5 - M3) = \mathbf{-0.002196}$
  - $\Delta AP(M5 - M6) = \mathbf{-0.001973}$

**Scientific Conclusion:**  
When the *flowers* category is excluded, M5's advantage over M3 and M6 vanishes and becomes slightly negative ($-0.0022$ and $-0.0020$). The apparent overall superiority of M5 is **disproportionately driven by a single category** (*flowers*), where colour dominates and adaptive weighting excels. Claims in the manuscript must state this transparently.

---

## 6. Learned M5 Weight Dispersion & Simplex Constraints

The MLP-predicted weights were inspected across all 100 queries:
- **Simplex Constraint Check:** $\min(w_C) \ge 0, \min(w_T) \ge 0$, and $|w_C + w_T - 1.0| \le 1.19 \times 10^{-7}$ across all queries (**100% compliant**).
- **Colour Weight ($w_C$):** Mean = **0.5784**, SD = **0.1568**, Min = **0.0956**, Max = **0.7523**.
- **Texture Weight ($w_T$):** Mean = **0.4216**, SD = **0.1568**, Min = **0.2477**, Max = **0.9044**.
- **Substantial Dispersion:** The standard deviation of $0.1568$ indicates that the MLP actively shifts modality emphasis based on query characteristics rather than collapsing onto fixed constants.

---

## 7. Artifact Manifest for Phase 5

The following deliverables have been generated in `output/phase5/`:
1. `output/phase5/results.json`: Full machine-readable metrics, hypothesis tests, bootstrap intervals, and sensitivity data.
2. `output/phase5/per_query_metrics.csv`: 600 query-level evaluations (100 queries $\times$ 6 models).
3. `output/phase5/category_metrics.csv`: 60 per-category aggregate records (10 categories $\times$ 6 models).
4. `output/phase5/query_weights.csv`: 100 query weight predictions for M5 and M6.
5. `output/phase5/rankings.npz`: Complete sorted gallery indices and distance matrices for all models.
6. `output/phase5/evaluation_report.md`: This comprehensive evaluation and audit document.
7. Figures:
   - `figures/phase5_overall_metrics_comparison.png`
   - `figures/phase5_category_map_breakdown.png`
   - `figures/phase5_query_difference_distribution.png`
   - `figures/phase5_flowers_influence_analysis.png`
   - `figures/phase5_m5_weight_distribution.png`
   - `figures/phase5_category_weights.png`

---

## 8. Compliance with Research Integrity Rules

- **No Data Fabrication:** Every reported number traces directly to machine-readable arrays and files.
- **No Overwriting:** Phase 1–4 artifacts remain completely untouched.
- **Hypothesis Testing Integrity:** Non-significant $p$-values are reported transparently without claiming statistical superiority.
- **Single-Dataset Boundary:** Generalization is strictly bounded to the evaluated Corel-1K queries.
