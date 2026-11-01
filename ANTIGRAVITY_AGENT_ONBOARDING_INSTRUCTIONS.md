# ANTIGRAVITY AGENT ONBOARDING & RESEARCH CONTEXT DIRECTIVE

> **MANDATORY INSTRUCTION FOR THE ANTIGRAVITY AGENT:**  
> If you have just been initialized in this repository (from a ZIP archive or Git clone), you are stepping into an active, high-rigor research project for **ICAIN 2026 (Springer LNNS, Paper ID 1567)**.  
> **DO NOT** assume this project is at an early stage.  
> **DO NOT** overwrite frozen results, re-split datasets, or retrain the model.  
> Your immediate task is to read this document, execute the mandatory verification walkthrough across all files, figures, tables, and documentation, and report your orientation status to the user.

---

## 1. Project Identity & Administrative Metadata

* **Paper Title:** *Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features*
* **Target Publication:** Springer Lecture Notes in Networks and Systems (LNNS)
* **Conference:** 7th International Conference on Artificial Intelligence and Networking (ICAIN 2026)
* **Paper ID:** 1567
* **GitHub Repository:** [https://github.com/Shalokexe/CBIR-Research-Paper.git](https://github.com/Shalokexe/CBIR-Research-Paper.git)
* **Current Status:** **Phases 1 through 10 COMPLETE, AUDITED, AND CRYPTOGRAPHICALLY VERIFIED.**
* **Unit Test Status:** **58 / 58 tests passing** (`pytest tests/`).

---

## 2. Mandatory First Actions: System Walkthrough Checklist

Before taking any user instructions or generating new content, you **MUST** execute the following inspection sequence:

### Step 1: Environment & Test Suite Verification
Run the complete automated unit test suite to verify codebase integrity:
```powershell
pytest tests/
```
*Verification standard: All 58 tests across `test_phase1.py` through `test_phase10.py` must pass cleanly.*

### Step 2: Inspect Manuscript Tables
Navigate to `manuscript_tables/` and verify the 7 publication-ready tables (available in both `.csv` and Springer `.tex` formats):
1. `table1_main_performance.{csv,tex}` — Corel-1K M1–M6 comparison.
2. `table2_statistical_significance.{csv,tex}` — Inferential Wilcoxon & Holm-Bonferroni tests.
3. `table3_category_metrics.{csv,tex}` — Per-category AP across all 10 classes.
4. `table4_gallery_sensitivity.{csv,tex}` — Nested gallery scaling (25%, 50%, 75%, 100%).
5. `table5_deep_baselines.{csv,tex}` — M5 vs LBP (D1), ResNet-50 (D2), EfficientNet (D3), CLIP (D5).
6. `table6_oxford_transfer.{csv,tex}` — Zero-shot Oxford-IIIT Pet transfer across 37 breeds.
7. `table7_runtime_scalability.{csv,tex}` — Latencies, throughput, and memory profiling.

### Step 3: Inspect 300 DPI Publication Figures
Navigate to `publication_figures/` and verify the 9 high-resolution manuscript figures:
1. `fig_system_architecture.png` — End-to-end system block diagram.
2. `fig_retrieval_pipeline.png` — 5-step query processing flow.
3. `fig_mlp_architecture.png` — Metric-learning neural network architecture (`13 -> 16 -> 8 -> 2`).
4. `fig_weight_distributions.png` — Query colour vs texture weight histograms.
5. `fig_main_performance_comparison.png` — M1–M6 Corel-1K benchmark bar chart.
6. `fig_category_performance.png` — Per-category Average Precision breakdown.
7. `fig_gallery_size_sensitivity.png` — Mathematical recall inflation under gallery reduction.
8. `fig_deep_learning_comparison.png` — Runtime, storage, and accuracy trade-off curves.
9. `fig_qualitative_retrieval_examples.png` — Visual query retrieval demonstration.

### Step 4: Review Core Audit & Documentation Reports
Inspect the following root-level reports:
- `README.md` — Master repository showcase documentation.
- `PROJECT_MASTER_CONTEXT_FOR_CLAUDE.txt` — Exhaustive plain-text reference summary.
- `final_research_audit.md` — 11-task comprehensive technical audit report.
- `reproducibility_checklist.md` — Machine learning reproducibility certification.
- `final_results_summary.md` — Master performance and statistical numbers.
- `manuscript_integration_recommendation.md` — Concrete section-by-section manuscript edits.
- `experiment_manifest.json` — SHA-256 cryptographic hashes of all deliverables.

---

## 3. Executive Technical Overview: What We Built

### The Core Problem
Heavy deep neural networks (ResNet-50, EfficientNet) and Vision-Language models (CLIP) achieve high retrieval accuracy, but their computational requirements (2,048–8,192 bytes/image, 10–25 ms extraction, heavy GPU/matrix ops) prohibit deployment on battery-powered edge devices, micro-servers, and embedded IoT hardware.

### The Proposed Solution (Proposed M5)
We designed, audited, and benchmarked an ultra-compact, query-adaptive hybrid CBIR framework:
1. **13-Dimensional Feature Representation:**
   * **9-D HSV Statistical Moments:** Mean ($\mu$), Standard Deviation ($\sigma$), and Skewness ($s$) across Hue, Saturation, and Value channels. Skewness uses an unbiased cube-root formulation with numerical epsilon guards.
   * **4-D GLCM Texture Moments:** Contrast, Dissimilarity, Homogeneity, Energy, Correlation, and ASM extracted across 4 angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) at distance $d=1$.
2. **Zero-Leakage Gallery Normalization:**
   * Features are Min-Max scaled strictly using gallery-derived bounds (`train_min`, `train_max`). Held-out queries are normalized using these pre-computed bounds. Zero test data was ever exposed.
3. **Adaptive Metric-Learning MLP:**
   * Architecture: `13 -> Linear(16) -> LeakyReLU(0.1) -> Linear(8) -> LeakyReLU(0.1) -> Linear(2) -> Softmax`.
   * Size: **378 trainable parameters (1.76 KB disk checkpoint)**.
   * Objective: Trained exclusively on 700 gallery training images using batch-hard triplet margin loss ($m=0.15$). Early stopping evaluated on 199 validation images.
   * Softmax output enforces the strict unit simplex constraint:
     $$w_C(q) + w_T(q) = 1.0, \quad w_C(q) > 0, \quad w_T(q) > 0$$
4. **Weighted Distance Ranking:**
   $$D_{M5}(q, g) = \sqrt{w_C(q) \cdot \sum_{i=1}^9 (f_{q,i}^C - f_{g,i}^C)^2 + w_T(q) \cdot \sum_{j=1}^4 (f_{q,j}^T - f_{g,j}^T)^2}$$

---

## 4. Master Experimental Findings & Frozen Numbers

All evaluations strictly follow the common benchmark protocol: **899 gallery images, 100 held-out queries (10 per category), and 1 near-duplicate purged (`horses/761.jpg`)**.

### Benchmark Performance (Corel-1K)
* **M1 (Colour Only):** P@10 = 0.6190, R@10 = 0.0689, F1@10 = 0.1240, mAP = 0.4045
* **M2 (Texture Only):** P@10 = 0.4680, R@10 = 0.0521, F1@10 = 0.0937, mAP = 0.3508
* **M3 (Equal Fusion 0.5/0.5):** P@10 = 0.6990, R@10 = 0.0778, F1@10 = 0.1400, mAP = 0.4650
* **M4 (Heuristic Fixed 0.7/0.3):** P@10 = 0.6730, R@10 = 0.0749, F1@10 = 0.1348, mAP = 0.4337
* **M5 (Proposed Adaptive):** P@10 = **0.7100**, R@10 = **0.0790**, F1@10 = **0.1422**, mAP = **0.4798**
* **M6 (Mean Weight Control):** P@10 = 0.6960, R@10 = 0.0774, F1@10 = 0.1394, mAP = 0.4605

*(Note: F1@10 is computed per query using unrounded precision and recall prior to averaging, avoiding the harmonic mean aggregation fallacy).*

### Deep-Learning Baselines & Edge Trade-Off (Phase 8)
* **D1 (Local Binary Patterns):** 10-D, Chi-Square, mAP = 0.3734, Storage = 40 B, Latency = 10.82 ms.
* **D2 (ResNet-50):** 2048-D, Cosine, mAP = 0.9103, Storage = 8,192 B, Latency = 24.04 ms.
* **D3 (EfficientNet-B0):** 1280-D, Cosine, mAP = 0.8879, Storage = 5,120 B, Latency = 9.71 ms.
* **D5 (CLIP ViT-B/32):** 512-D, Cosine, mAP = 0.9485, Storage = 2,048 B, Latency = 16.60 ms.
* **Proposed M5 Niche:** While deep models dominate semantic accuracy ($0.88 - 0.95$ mAP), M5 requires **$157.5\times$ less storage** (52 bytes/image), extracts **$53.4\times$ faster on CPU** (0.45 ms/image), and uses a **1.76 KB** model footprint.

### Zero-Shot Cross-Dataset Transfer (Oxford-IIIT Pet, Phase 9)
* Tested on 7,390 images across 37 cat/dog breeds without retraining or fine-tuning.
* Results: M5 mAP = **0.033332**, M6 mAP = 0.033155, M3 mAP = 0.033275.
* Random chance baseline: $1/37 \approx 0.0270$.
* Proves the domain boundary: global colour and texture moments cannot distinguish fine-grained domestic biological breeds without deep local spatial geometry.

---

## 5. Mandatory Research Integrity & Statistical Invariants

The revised manuscript adheres to the highest standards of scientific honesty. You must maintain these disclosures:
1. **Statistically Significant Contrasts:**
   * M5 statistically significantly outperforms classical LBP (D1, $p = 6.61 \times 10^{-9}$) and heuristic fixed fusion (M4, $p = 5.35 \times 10^{-8}$).
2. **Descriptive-Only Contrasts:**
   * Differences between M5 and Equal Fusion M3 ($+0.0148$) or Mean Control M6 ($+0.0193$) are **not statistically significant** after Holm-Bonferroni correction ($p = 0.6519$).
3. **Category Driver Effect (Flowers):**
   * M5's advantage over static controls is driven primarily by the Flowers category ($+0.2103$ mAP gain over M6). Excluding Flowers, the mean paired difference across the remaining 9 classes is $\Delta = -0.00197$.
4. **Gallery Scaling Artifact:**
   * In Phase 7 (C2), shrinking gallery size from 899 to 220 images increases Recall@10 from 0.0790 to 0.2882 and F1@10 from 0.1422 to 0.3662. This is a mathematical artifact of a smaller relevance denominator, not superior retrieval fidelity.
5. **No False Claims of Universal Superiority:**
   * Do not claim M5 outperforms deep learning. M5 is an edge-optimized handcrafted solution for compute-constrained hardware.

---

## 6. What Needs to Be Done Next (Current Task Scope)

The experimental code, benchmarks, statistical analyses, tables, and figures are 100% complete and frozen.  
The current task is **Manuscript Integration & Documentation**:
1. Review [manuscript_integration_recommendation.md](manuscript_integration_recommendation.md) for precise instructions on updating the paper draft.
2. Update the conference manuscript (LaTeX/Word format) using the verified tables in `manuscript_tables/` and figures in `publication_figures/`.
3. Ensure all text claims match the audited numbers in [final_results_summary.md](final_results_summary.md) and [final_research_audit.md](final_research_audit.md).

---

## 7. Strict Rules for the Antigravity Agent

* **RULE 1 — NEVER MODIFY EXPERIMENTAL RESULTS:** Do not alter numbers in `results.json`, `manuscript_tables/`, or output CSVs.
* **RULE 2 — NEVER RETRAIN THE MODEL:** The checkpoint `model/adaptive_weight_mlp.pt` is audited and frozen.
* **RULE 3 — MAINTAIN DOCUMENTATION INTEGRITY:** Preserve existing comments, docstrings, and audit trails.
* **RULE 4 — PRESERVE UNIT TESTS:** Any newly added scripts or helpers must maintain 100% pass rate on `pytest tests/`.
* **RULE 5 — USE FILE LINKS:** Always provide clickable `file:///` markdown links with forward slashes when referring to project files.

