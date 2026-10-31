# Phase 9 — External Transfer Analysis Report: Oxford-IIIT Pet
**Project:** ICAIN 2026, Springer LNNS — Paper ID 1567  
**Title:** Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features  
**Focus:** Zero-Shot Cross-Dataset Transferability & Domain Shift Evaluation  
**Date:** 2026-11-01 02:04:06  

---

## 1. Executive Summary & Scientific Findings

This report evaluates the **zero-shot external transferability** of the proposed adaptive CBIR system (**Proposed M5**) when applied without retraining to the **Oxford-IIIT Pet** benchmark.

### Core Empirical Conclusions:
1. **Low Absolute Transfer Performance (Fine-Grained Domain Shift):**
   - On Oxford-IIIT Pet, **Proposed M5 achieves mAP = 0.033332** and **P@10 = 0.0546**.
   - The fixed mean weight control **M6 achieves mAP = 0.033155**, while Equal Fusion **M3 achieves mAP = 0.033275**.
   - These low numerical values (~3.38%) reflect the fundamental nature of fine-grained pet classification across **37 breeds**, where random retrieval chance is $1/37 \approx 0.0270$ (2.70%).
2. **Reconciliation of Manuscript Discrepancies:**
   - Previous versions of the manuscript PDF reported conflicting mAP numbers ($0.033832$ vs $0.033882$).
   - Our standardized audit establishes that on Oxford-IIIT Pet, the difference between M5 and M6 is **$\Delta = +0.000177$** (less than 0.02% mAP).
3. **Mandatory Methodological Invariant:**
   - **Do NOT claim universal cross-dataset generalisation.** Small numerical differences on Oxford-IIIT Pet do not establish cross-dataset superiority. 
   - Global handcrafted colour-texture descriptors are inherently limited on fine-grained pet breeds where colour varies within breeds and overlaps across distinct breeds.

---

## 2. External Dataset Specification

| Parameter | Specification |
|---|---|
| **Dataset Name** | Oxford-IIIT Pet Dataset |
| **Version** | 1.0 (Parkhi et al., CVPR 2012) |
| **Source URL** | https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| **License** | Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) |
| **Total Images** | 7390 JPEG images |
| **Category / Breed Count** | 37 fine-grained breeds (12 Cats, 25 Dogs) |
| **Query Protocol** | 370 held-out queries (10 per breed, exactly mirroring Corel-1K protocol) |
| **Gallery Protocol** | 7020 gallery images (strictly disjoint from queries) |
| **Normalization** | Gallery-fitted Min-Max normalization (zero query leakage) |
| **Transfer Protocol** | **Strict Zero-Shot Transfer**: Frozen MLP checkpoint (`adaptive_weight_mlp.pt`) trained exclusively on Corel-1K; zero fine-tuning on Pet images |

---

## 3. Comparative Transfer Performance Matrix

| Model Configuration | Feature Dim | Precision@10 | Recall@10 | F1@10 | Mean AP (mAP) | Ranking Latency |
|---|---|---|---|---|---|---|
| `M1_Colour` | 9 | 0.0562 | 0.0030 | 0.0056 | **0.031997** | 0.255 ms |
| `M2_Texture` | 4 | 0.0338 | 0.0018 | 0.0034 | **0.031960** | 0.111 ms |
| `M3_Equal` | 13 | 0.0568 | 0.0030 | 0.0057 | **0.033275** | 0.127 ms |
| `M4_Fixed` | 13 | 0.0570 | 0.0030 | 0.0057 | **0.032556** | 0.132 ms |
| `M5_Proposed_Adaptive` | 13 | 0.0546 | 0.0029 | 0.0055 | **0.033332** | 0.132 ms |
| `M6_Mean_Control` | 13 | 0.0584 | 0.0031 | 0.0059 | **0.033155** | 0.132 ms |

---

## 4. Species Breakdown: Cats vs. Dogs

| Species | Breed Count | M1 (Colour) mAP | M2 (Texture) mAP | M3 (Equal) mAP | M5 (Proposed) mAP | M6 (Mean Ctrl) mAP |
|---|---|---|---|---|---|---|
| **Cats** | 12 | 0.0294 | 0.0332 | 0.0319 | **0.0326** | 0.0315 |
| **Dogs** | 25 | 0.0332 | 0.0314 | 0.0339 | **0.0337** | 0.0339 |

---

## 5. Scientific Discussion for Manuscript Revision

- **Coarse-Grained vs. Fine-Grained Retrieval:** The Corel-1K dataset consists of distinct semantic categories (e.g. Dinosaurs vs Mountains vs Buses), where global colour histograms and GLCM texture provide strong discriminatory boundaries. In contrast, Oxford-IIIT Pet is a fine-grained task requiring localized facial/ear geometry and subtle coat patterns. Global colour/texture moments cannot disambiguate a Beagle from an English Foxhound.
- **Reporting Stance:** In the revised manuscript, the Oxford-IIIT Pet experiment should be framed transparently as a boundary condition highlighting the limitations of low-dimensional handcrafted descriptors in fine-grained transfer, rather than claiming cross-dataset superiority.
