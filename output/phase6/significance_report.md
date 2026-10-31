# Phase 6 — Comprehensive Statistical Validation Report

**Conference:** ICAIN 2026, Springer LNNS — Paper ID 1567  
**Title:** Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features  
**Evaluation Scope:** Inferential Statistical Testing, Multiple-Comparison Corrections, Effect Sizes, and Sensitivity Analysis of Frozen Phase 5 Retrieval Results  
**Date:** 2026-11-01 00:33:47  

---

## 1. Executive Summary & Inferential Verdict

This report presents a separate, rigorous inferential statistical analysis of the frozen retrieval results produced in Phase 5. In strict accordance with experimental research integrity rules, **no models were retrained, no baselines were modified, and no parameters were altered**. All statistical metrics are computed directly from the saved query-level performance records.

### Key Inferential Findings:
1. **No Statistically Significant Superiority:** Under the prespecified family of primary hypotheses, the proposed query-adaptive model (**M5**) is **not statistically significantly superior** to either the fixed distance-level control (**M6**) or the equal-fusion baseline (**M3**) under two-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction:
   - **M5 vs M6:** Wilcoxon $W = 2353.0$, raw $p = 0.5543$, **Holm-adjusted $p = 0.6519$** (Non-significant).
   - **M5 vs M3:** Wilcoxon $W = 1972.0$, raw $p = 0.3259$, **Holm-adjusted $p = 0.6519$** (Non-significant).
2. **Evenly Balanced Query Wins and Losses:** Across the 100 evaluated held-out queries:
   - Against **M3**: M5 achieves **51 wins, 6 ties, and 43 losses** ($pprox 51\%$ win rate).
   - Against **M6**: M5 achieves **53 wins, 0 ties, and 47 losses** ($pprox 53\%$ win rate).
   This near-even split across heterogeneous queries explains why rank-based non-parametric tests do not reject the null hypothesis of equal performance.
3. **The Flowers Category Outlier Effect:** The aggregate descriptive mAP advantage of M5 ($+0.0148$ vs M3 and $+0.0193$ vs M6) is almost entirely driven by a single category (**Flowers**):
   - In Flowers alone, M5 outperforms M3 by **$+0.1673$** and M6 by **$+0.2103$**.
   - Across the remaining 9 categories combined (90 queries), M5 slightly underperforms both baselines: **$-0.0022$** vs M3 and **$-0.0020$** vs M6.
4. **Bootstrap vs Wilcoxon Inferential Distinction:** While the 95% stratified bootstrap confidence intervals for the sample mean difference are strictly positive ($[0.0074, 0.0234]$ for M5 vs M3 and $[0.0119, 0.0280]$ for M5 vs M6), they assess the *population arithmetic mean* (which is pulled positive by the large magnitude of flower gains). In contrast, the Wilcoxon test assesses the *median/rank sign distribution*. A positive bootstrap confidence interval does **not** make the Wilcoxon test significant.

---

## 2. Prespecified Hypothesis Testing Framework

To control the family-wise error rate (FWER) at alpha = 0.05, two primary orthogonal comparisons were prespecified:
- **Hypothesis 1 (H0,1):** AP(M5) = AP(M6) (Query-adaptive distance fusion vs fixed reference mean query weights).
- **Hypothesis 2 (H0,2):** AP(M5) = AP(M3) (Query-adaptive distance fusion vs equal feature concatenation).

Secondary exploratory comparisons (M5 vs M1, M5 vs M2, and M5 vs M4) and sensitivity subsets excluding Flowers were analyzed separately.

---

## 3. Comprehensive Paired Inferential Test Results

The table below summarizes all statistical tests conducted on the 100 paired query Average Precision values:

| Comparison | Status | Mean Diff ($ar{d}$) | Median Diff | Std Dev ($s_d$) | Wins / Ties / Losses | Wilcoxon $W$ | Wilcoxon Raw $p$ | Holm-Adjusted $p$ | Cohen's $d_z$ | Rank-Biserial $r_B$ | Permutation Raw $p$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M5 vs M6** | **Primary** | **+0.019258** | +0.000214 | 0.076450 | **53 / 0 / 47** | 2353.0 | 0.554258 | **0.651876** | 0.2519 | +0.0681 | 0.006160 |
| **M5 vs M3** | **Primary** | **+0.014758** | +0.000158 | 0.065948 | **51 / 6 / 43** | 1972.0 | 0.325938 | **0.651876** | 0.2238 | +0.1167 | 0.018580 |
| **M5 vs M4** | Secondary | +0.046135 | +0.003929 | 0.089851 | 63 / 1 / 36 | 1341.0 | 0.001460 | — | 0.5135 | +0.4582 | 0.000020 |
| **M5 vs M1** | Secondary | +0.075332 | +0.045053 | 0.125587 | 74 / 0 / 26 | 933.0 | 0.000004 | — | 0.5998 | +0.6305 | 0.000000 |
| **M5 vs M2** | Secondary | +0.128978 | +0.106516 | 0.174151 | 75 / 0 / 25 | 711.0 | 0.000000 | — | 0.7406 | +0.7184 | 0.000000 |
| **M5 vs M6 (No Flowers)** | Sensitivity | -0.001973 | -0.000504 | 0.013899 | 43 / 0 / 47 | 1742.0 | 0.218983 | — | -0.1419 | -0.1482 | 0.181778 |
| **M5 vs M3 (No Flowers)** | Sensitivity | -0.002196 | +0.000000 | 0.022082 | 41 / 6 / 43 | 1613.0 | 0.443033 | — | -0.0995 | -0.0887 | 0.350126 |

---

## 4. Category-Stratified Paired Bootstrap Confidence Intervals

Because Corel-1K queries are structured by category (10 queries per category), an unstratified bootstrap would distort the semantic class balance. We applied a **category-stratified paired bootstrap** with 10,000 resamples (seed 42), resampling 10 queries with replacement within each category:

| Comparison | Point Estimate | 95% Percentile CI | 95% BCa CI | 99% Percentile CI | Bootstrap SE | Zero Cross? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **M5 vs M6** | **+0.019258** | **[0.011935, 0.027971]** | [0.007654, 0.039509] | [0.009594, 0.031163] | 0.004142 | No |
| **M5 vs M3** | **+0.014758** | **[0.007414, 0.023385]** | [0.004804, 0.032261] | [0.005162, 0.026592] | 0.004101 | No |
| **M5 vs M4** | +0.046135 | [0.035763, 0.056757] | [0.027876, 0.073773] | [0.032554, 0.060151] | 0.005345 | No |
| **M5 vs M1** | +0.075332 | [0.060909, 0.089853] | [0.051977, 0.107656] | [0.056067, 0.094110] | 0.007446 | No |
| **M5 vs M2** | +0.128978 | [0.102938, 0.155300] | [0.101482, 0.156732] | [0.095219, 0.164144] | 0.013281 | No |
| **M5 vs M6 (No Flowers)** | -0.001973 | [-0.004560, 0.000536] | [-0.005032, 0.000706] | [-0.005335, 0.001276] | 0.001289 | **Yes (Crosses 0)** |
| **M5 vs M3 (No Flowers)** | -0.002196 | [-0.006356, 0.001956] | [-0.006929, 0.002241] | [-0.007486, 0.003141] | 0.002102 | **Yes (Crosses 0)** |

---

## 5. Detailed Breakdown of the Flowers Outlier Effect

The table below breaks down the paired differences category-by-category:

| Semantic Category | Query Count | M5 vs M6 Mean $\Delta AP$ | M5 vs M3 Mean $\Delta AP$ | Category Leading Model |
|:---|:---:|:---:|:---:|:---:|
| `flowers` | 10 | **+0.210336** | **+0.167349** | **M5** |
| `dinosaurs` | 10 | +0.005518 | +0.000668 | **M5** |
| `elephants` | 10 | +0.005210 | +0.009656 | **M5** |
| `beaches` | 10 | +0.000505 | +0.002508 | **M5** |
| `monuments` | 10 | +0.000169 | +0.001475 | **M5** |
| `horses` | 10 | -0.000537 | +0.002351 | **M6** |
| `peolpe_and_villages_in_Africa` | 10 | -0.000785 | +0.000196 | **M6** |
| `foods` | 10 | -0.004715 | -0.006751 | **M3** |
| `mountains_and_snow` | 10 | -0.006638 | -0.005292 | **M6** |
| `bus` | 10 | -0.016482 | -0.024579 | **M3** |

### Mathematical Decomposition:
- The full 100-query mean difference for M5 vs M6 is:
  $$ar{d} = 0.10 	imes (+0.210336) + 0.90 	imes (-0.001973) = +0.021034 - 0.001776 = +0.019258$$
- The Flowers category alone accounts for **$109.2\%$** of the net positive difference in M5 vs M6 and **$113.4\%$** in M5 vs M3!
- When Flowers is excluded, the permutation $p$-value jumps from $0.0062$ to **$0.1818$**, and the Wilcoxon $p$-value remains non-significant at **$0.2190$**.

---

## 6. Methodological and Inferential Limitations

1. **Common Gallery Conditioning:** All paired comparisons share the identical 899-image gallery. Consequently, observations are statistically dependent on this specific gallery composition. The inferential tests assess variance across queries conditional on this gallery, not variance across independent gallery realisations.
2. **Single-Dataset Boundary:** The evaluation is conducted on Corel-1K. Statistical superiority cannot be generalized to broader, heterogeneous image databases.
3. **Clarification on Sensitivity Experiments:** Earlier exploratory experiments testing five different train/val/query splits represent sensitivity to split selection, not five independent training runs on a fixed partition. They must not be conflated with independent replication.

---

## 7. Manuscript Writing Recommendations

When preparing the final manuscript for Springer LNNS:
- **Do not write:** *"M5 achieves statistically significant improvements over all baselines."* (This is false; Holm-adjusted $p = 0.6519$).
- **Write:** *"While M5 achieved the highest descriptive mAP (0.4798 vs 0.4650 for M3 and 0.4605 for M6), paired two-sided Wilcoxon signed-rank tests across the 100 held-out queries indicate that these differences are not statistically significant after Holm-Bonferroni correction ($p = 0.6519$). M5's performance gains were concentrated in visually distinctive colour-dominant classes (notably flowers), whereas fixed fusion performed comparably or slightly better in more heterogeneous categories."*
