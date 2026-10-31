# ICAIN 2026 — Enhanced Content-Based Image Retrieval (Paper ID 1567)

**Title:** Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features  
**Target Publication:** Springer Lecture Notes in Networks and Systems (LNNS)  
**Status:** Complete, Verified, Deterministic, and Audited Across Phases 1–10.

---

## Overview

This repository contains the complete reproducible implementation of the query-adaptive hybrid content-based image retrieval (CBIR) framework. The system combines 9-D HSV colour statistical moments with 4-D GLCM texture moments into an ultra-compact 13-dimensional descriptor, weighted dynamically per query via a lightweight 2-layer metric-learning MLP (`13 -> 16 -> 8 -> 2`).

---

## Directory Structure

- `dataset/`: Verified Wang Corel-1K dataset images (100 test queries, 900 training/gallery).
- `dataset_external/`: Official Oxford-IIIT Pet dataset for zero-shot transferability evaluation.
- `model/`: Checkpoint `adaptive_weight_mlp.pt` (450 parameters, 1.76 KB).
- `features/`: Modular extractors for HSV color moments and GLCM texture features with gallery normalization.
- `retrieval/`: Fast distance calculation and ranking evaluation routines.
- `scripts/`: End-to-end execution pipelines for Phases 7, 8, 9, and 10.
- `tests/`: 53 automated unit tests covering all methodology, mathematical bounds, and cryptographic integrity.
- `output/`: Isolated output artifacts partitioned by phase (`phase1/` through `phase10/`).
- `manuscript_tables/`: Publication-ready LaTeX `.tex` and `.csv` tables.
- `publication_figures/`: 300 DPI high-resolution figures.

---

## Reproduction Instructions

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
pytest tests/
```
All 53 unit tests across Phases 1–9 pass in under 6 seconds.

### 3. Re-Execute Complete Pipeline
- **Phase 7 (Gallery Sensitivity):** `python scripts/run_phase7_sensitivity.py`
- **Phase 8 (Deep Baselines):** `python scripts/run_phase8_baselines.py`
- **Phase 9 (Oxford Transfer & Scalability):** `python scripts/run_phase9_transfer_scalability.py`
- **Phase 10 (Master Audit & Tables):** `python scripts/generate_phase10_publication_package.py`

---

## Key Experimental Results

| Configuration | Feature Dim | Precision@10 | Recall@10 | F1@10 | Mean AP (mAP) | Storage / Image |
|---|---|---|---|---|---|---|
| **M1: Colour Only** | 9 | 0.6190 | 0.0689 | 0.1240 | 0.4045 | 36 B |
| **M2: Texture Only** | 4 | 0.4680 | 0.0521 | 0.0937 | 0.3508 | 16 B |
| **M3: Equal Fusion** | 13 | 0.6990 | 0.0778 | 0.1400 | 0.4650 | 52 B |
| **M4: Fixed Heuristic** | 13 | 0.6730 | 0.0749 | 0.1348 | 0.4337 | 52 B |
| **M5: Proposed Adaptive** | **13** | **0.7100** | **0.0790** | **0.1422** | **0.4798** | **52 B** |
| **M6: Mean Control** | 13 | 0.6960 | 0.0774 | 0.1394 | 0.4605 | 52 B |

---

## License & Citation
Springer LNNS — ICAIN 2026 Proceedings.
