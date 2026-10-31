# Phase 4 Training Provenance and Checkpoint Audit Report

**Project:** ICAIN 2026, Springer LNNS — Paper ID 1567  
**Title:** Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features  
**Auditor:** Antigravity Pair Programming Agent  
**Date:** 2026-10-31  
**Audit Target:** Phase 4 MLP Checkpoint, Training Provenance, Data Splits, Inference Determinism, and M1–M6 Definitions  

---

## 1. Executive Summary & Verdict

| Audit Dimension | Status | Key Finding |
|---|---|---|
| **MLP Checkpoint Integrity** | **VERIFIED** | Checkpoint `model/adaptive_weight_mlp.pt` is a valid PyTorch v2 zip archive with all 378 model parameters intact. Rich metadata dictionary found inside `m5_checkpoint/data.pkl`. |
| **Data Partition & Isolation** | **VERIFIED** | Exact 700 train / 199 val / 100 query / 1 excluded split. Checkpoint `train_indices` and `val_indices` match `output/phase1/split_manifest.csv` and `features.npz` 100%. Queries were strictly isolated. |
| **Inference Reproducibility** | **VERIFIED** | Forward pass on frozen Phase 2 query features matches saved `output/phase4/query_weights.csv` float-for-float (max error: $5.96 \times 10^{-8}$). |
| **M6 Control Definition** | **VERIFIED** | M6 weights ($w_C = 0.578404, w_T = 0.421596$) match the exact empirical mean of M5 predictions over the 100 query features. Reference constants in `config.py` match perfectly. |
| **Training Source Code** | **MISSING IN SNAPSHOT** | **Provenance Gap Confirmed:** No training script (e.g. `train_mlp.py`), triplet sampler, or loss module is present in the repository snapshot. Training cannot be independently re-run from code alone without reconstructing the trainer. |
| **Methodology Text Alignment** | **STALE IN DOCS** | `docs/extracted_methodology.txt` refers to obsolete 900-gallery / 200-val counts. The actual code, checkpoint metadata, and outputs use the corrected 899-gallery / 199-val protocol. |

**Audit Verdict:**  
The saved Phase 4 retrieval outputs (`output/phase4/rankings.npz`, `query_weights.csv`, `per_query_metrics.csv`, `aggregate_metrics.json`) are **fully verified, mathematically consistent, and deterministically generated** from the frozen Phase 2 features and the saved MLP checkpoint. However, **training reproducibility is incomplete** because the script used to generate `model/adaptive_weight_mlp.pt` was omitted from this ZIP snapshot.

---

## 2. Deep Inspection of `model/adaptive_weight_mlp.pt`

### 2.1 File & Archive Structure
Inspection of `model/adaptive_weight_mlp.pt` reveals a standard PyTorch ZIP container:
- `m5_checkpoint/data.pkl` (4,905 bytes): Python pickle containing the metadata dictionary and state dict mapping.
- `m5_checkpoint/data/0` (832 bytes): `fc1.weight`, shape `(16, 13)` in `float32` (208 floats $\times$ 4 bytes = 832 bytes).
- `m5_checkpoint/data/1` (64 bytes): `fc1.bias`, shape `(16,)` in `float32` (16 floats $\times$ 4 bytes = 64 bytes).
- `m5_checkpoint/data/2` (512 bytes): `fc2.weight`, shape `(8, 16)` in `float32` (128 floats $\times$ 4 bytes = 512 bytes).
- `m5_checkpoint/data/3` (32 bytes): `fc2.bias`, shape `(8,)` in `float32` (8 floats $\times$ 4 bytes = 32 bytes).
- `m5_checkpoint/data/4` (64 bytes): `fc3.weight`, shape `(2, 8)` in `float32` (16 floats $\times$ 4 bytes = 64 bytes).
- `m5_checkpoint/data/5` (8 bytes): `fc3.bias`, shape `(2,)` in `float32` (2 floats $\times$ 4 bytes = 8 bytes).

Total trainable parameters: **378 parameters**.

### 2.2 Embedded Checkpoint Metadata
Extracting `m5_checkpoint/data.pkl` via safe unpickling revealed an extensive, previously undocumented metadata payload saved by the original trainer:

```json
{
  "checkpoint_format": 2,
  "feature_fingerprint": "842ca497932b388101e9d23220045f1b13ecff0944e2896259e20a7c5b4e9715",
  "model_source_fingerprint": "ccfdbf501148034ff96dd70fde6a93c514318ccdf6490ce2bc02cd86c0858f05",
  "split_fingerprint": "1026fcf2ac7ce6942a8cd4490d63000430ec10488dc2f0d6d52b11381c8c0b58",
  "split_manifest_sha256": "cee8d17765467bbea92fd6654b38e1c9f06e8bf0f3cea246890019ec743efc71",
  "training_config": {
    "training_protocol_version": 3,
    "seed": 42,
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.001,
    "margin": 0.15,
    "patience": 10,
    "train_per_category": 70,
    "validation_per_category": 20,
    "positive_samples_per_query": 5,
    "negative_samples_per_query": 10,
    "validation_positive_samples_per_query": 2,
    "validation_negative_samples_per_query": 5,
    "model_dimensions": [13, 16, 8, 2],
    "split_mode": "explicit_group_disjoint",
    "train_count": 700,
    "validation_count": 199
  },
  "split_info": {
    "train_indices": "<list of 700 integer indices in [0, 898]>",
    "val_indices": "<list of 199 integer indices in [0, 898]>",
    "best_epoch": 8,
    "best_val_loss": 0.061751868452259044
  }
}
```

### 2.3 Training Loss & Early Stopping Trajectory
The metadata contains the exact 18-epoch training and validation loss progression:

| Epoch | Train Loss | Validation Loss | Notes |
|:---:|:---:|:---:|:---|
| 1 | 0.067390 | 0.061889 | Initial epoch |
| 2 | 0.066065 | 0.062029 | |
| 3 | 0.065750 | 0.061784 | |
| 4 | 0.065578 | 0.061764 | |
| 5 | 0.065451 | 0.062360 | |
| 6 | 0.065326 | 0.061953 | |
| 7 | 0.065214 | 0.062754 | |
| **8** | **0.065096** | **0.061752** | **Best Validation Loss (Checkpoint selected)** |
| 9 | 0.064872 | 0.062916 | Patience 1 |
| 10 | 0.064771 | 0.062302 | Patience 2 |
| 11 | 0.064648 | 0.062221 | Patience 3 |
| 12 | 0.064469 | 0.062351 | Patience 4 |
| 13 | 0.064397 | 0.062562 | Patience 5 |
| 14 | 0.064275 | 0.062665 | Patience 6 |
| 15 | 0.064153 | 0.062557 | Patience 7 |
| 16 | 0.064071 | 0.063699 | Patience 8 |
| 17 | 0.063962 | 0.062849 | Patience 9 |
| 18 | 0.063888 | 0.063236 | Patience 10 $\rightarrow$ **Early stopping triggered** |

**Confirmation:** Epoch 8 was indeed the optimal checkpoint selected by early stopping on validation loss, exactly confirming the master prompt reference.

---

## 3. Training Provenance Gap Analysis

### 3.1 What is Present
1. `model/adaptive_weight_mlp.pt` with trained parameter tensors and complete hyperparameter/epoch history metadata.
2. `model/mlp_model.py` containing model class definition `AdaptiveWeightMLP` and checkpoint loader `load_trained_mlp`.
3. `config.py` declaring architecture dims, batch size 64, lr 0.001, margin 0.15, patience 10, positive/negative sample counts.
4. `scripts/run_phase4_retrieval.py` performing retrieval evaluation and metrics computation.

### 3.2 What is Missing
1. **Training Script:** No Python script exists that executes the training loop (e.g. `train_mlp.py` or `scripts/run_phase4_training.py`).
2. **Triplet Loss & Sampler Code:** No implementation exists in code for generating triplet batches $(q, p, n)$, nor is there active code computing $L = \max(0, D_{M5}^2(q, p) - D_{M5}^2(q, n) + 0.15)$.
3. **Execution Logs:** No raw terminal stdout log or tensorboard run from the training execution was saved in the workspace.

### 3.3 Provenance Implication
While the saved checkpoint metadata and Phase 4 retrieval metrics are internally consistent, **we cannot independently retrain the model from scratch** using only the files currently in this snapshot. Any claim in the research paper about full end-to-end reproducibility must acknowledge that the training script is a companion deliverable to be reconstructed or imported.

---

## 4. Dataset Partition and Leakage Audit

### 4.1 Split Manifest Verification
- File audited: `output/phase1/split_manifest.csv`
- Actual SHA256: `cee8d17765467bbea92fd6654b38e1c9f06e8bf0f3cea246890019ec743efc71`
- Checkpoint `split_manifest_sha256`: `cee8d17765467bbea92fd6654b38e1c9f06e8bf0f3cea246890019ec743efc71`
- **Result:** Exact match. The checkpoint was trained against this identical split manifest file.

### 4.2 Gallery and Query Partition Counts
- Total entries in manifest: 1,000 images
  - `mlp_train`: 700 images (70 per category $\times$ 10 categories)
  - `mlp_validation`: 199 images (20 per category for 9 categories; 19 for horses)
  - `held_out_query`: 100 images (10 per category $\times$ 10 categories)
  - `excluded_near_duplicate_of_query`: 1 image (`training_set/horses/761.jpg`)
- Evaluated gallery size: $700 + 199 = 899$ images.
- Evaluated queries: 100 images.

### 4.3 Training/Validation Index Correspondence
Checking the 700 `train_indices` and 199 `val_indices` stored in the checkpoint against the gallery images in `output/phase2/features.npz`:
- `ckpt_train_idx == npz_train_idx`: **TRUE** (all 700 indices match `mlp_train`).
- `ckpt_val_idx == npz_val_idx`: **TRUE** (all 199 indices match `mlp_validation`).
- `ckpt_train_idx ∩ ckpt_val_idx`: **EMPTY SET** (0 overlap).
- Query isolation: Query images occupy indices 899 through 998 in `features.npz`. None of these indices appear in `train_indices` or `val_indices`.
- Normalization isolation: MinMax parameters stored in the checkpoint (`color_min`, `color_max`, `texture_min`, `texture_max`) match `output/phase2/feature_metadata.json` exactly to float32 precision, which was fitted on the 899 gallery images only.
- **Result:** Strict query isolation is verified. No data leakage occurred.

---

## 5. Model Weight Verification and Inference Determinism

To independently verify the checkpoint without PyTorch installed in the local environment, a standalone forward inference engine was executed using pure NumPy on the frozen Phase 2 query features (`q_feat` shape `(100, 13)`):

$$\mathbf{h}_1 = \text{LeakyReLU}_{0.1}(\mathbf{x} \mathbf{W}_1^T + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \text{LeakyReLU}_{0.1}(\mathbf{h}_1 \mathbf{W}_2^T + \mathbf{b}_2)$$
$$\mathbf{z} = \mathbf{h}_2 \mathbf{W}_3^T + \mathbf{b}_3$$
$$[w_C, w_T] = \text{Softmax}(\mathbf{z})$$

### Numerical Parity Test
Comparing pure NumPy forward predictions with `output/phase4/query_weights.csv`:
- Maximum absolute difference: $5.9605 \times 10^{-8}$ (pure float32 precision difference)
- Mean absolute difference: $1.6987 \times 10^{-8}$
- Simplex constraints:
  - $\min(w_C) = 0.095598 \ge 0$
  - $\min(w_T) = 0.247656 \ge 0$
  - $|w_C + w_T - 1.0| \le 1.19 \times 10^{-7}$ across all 100 queries.

### Weight Dispersion Statistics
- Colour weight $w_C$: Mean = **0.578404**, SD = **0.156828**, Range = [0.0956, 0.7523]
- Texture weight $w_T$: Mean = **0.421596**, SD = **0.156828**, Range = [0.2477, 0.9044]

The non-zero variance ($SD \approx 0.1568$) confirms that the model actively adapts weights per query rather than degenerating into a static constant.

---

## 6. M6 Fixed Control Definition Verification

The definition and constants for M6 were audited:
- Empirical mean of M5 predicted weights over 100 queries:
  - $\bar{w}_C = 0.578403890132904$
  - $\bar{w}_T = 0.421596109867096$
- Saved M6 weights in `output/phase4/query_weights.csv`:
  - `m6_color_weight` = $0.578403890132904$
  - `m6_texture_weight` = $0.421596109867096$
- Values in `config.py`:
  - `M6_REFERENCE_COLOR_WEIGHT = 0.5784038992226124`
  - `M6_REFERENCE_TEXTURE_WEIGHT = 0.42159609973430634`

**Conclusion:** M6 is an exact distance-level fixed control using the reference query mean weights. It is **not** an ex-ante independent model or a validation-selected parameter. This distinction must be preserved in all paper writeups.

---

## 7. M1–M6 Performance & Parity Summary

Cross-referencing `output/phase4/aggregate_metrics.json` against the reference values:

| Model | Fusion Type | Metric Space | P@10 | R@10 | F1@10 | mAP | Reference mAP | Match Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1** | Single modality | Colour 9D | 0.6190 | 0.068890 | 0.123982 | 0.404462 | 0.404462 | **Exact Match** |
| **M2** | Single modality | Texture 4D | 0.4680 | 0.052067 | 0.093709 | 0.350816 | 0.350816 | **Exact Match** |
| **M3** | Feature-level | Concatenation $[0.5C, 0.5T]$ | 0.6990 | 0.077779 | 0.139982 | 0.465036 | 0.465036 | **Exact Match** |
| **M4** | Feature-level | Concatenation $[0.7C, 0.3T]$ | 0.6730 | 0.074893 | 0.134786 | 0.433659 | 0.433659 | **Exact Match** |
| **M5** | Distance-level | Adaptive $w_C(q) D_C^2 + w_T(q) D_T^2$ | **0.7100** | **0.079000** | **0.142180** | **0.479794** | **0.479794** | **Exact Match** |
| **M6** | Distance-level | Fixed Control $\bar{w}_C D_C^2 + \bar{w}_T D_T^2$ | 0.6960 | 0.077443 | 0.139378 | 0.460536 | 0.460536 | **Exact Match** |

Mathematical distinction confirmed:
- M3/M4 squared distance contribution: $w^2 \cdot D^2$ ($0.25 D_C^2 + 0.25 D_T^2$ and $0.49 D_C^2 + 0.09 D_T^2$).
- M5/M6 squared distance contribution: $w \cdot D^2$ ($w_C D_C^2 + w_T D_T^2$).

---

## 8. Inspected Files Inventory

The following files were opened, analyzed, and audited during this provenance review:

1. **Handover & Rules:**
   - `handover/00_START_HERE.md`
   - `handover/01_PROJECT_CONTEXT_MASTER.md`
   - `handover/02_AGENT_RULES.md`
   - `handover/03_PHASE_STATUS.md`
   - `handover/04_NEXT_AGENT_PROMPT.md`
   - `handover/OUTPUT_REGISTER.csv`
   - `handover/README.md`
2. **Configuration & Method Documentation:**
   - `config.py`
   - `docs/extracted_methodology.txt`
3. **Model & Architecture:**
   - `model/adaptive_weight_mlp.pt` (full binary and zip archive inspection, pickle opcode disassembly, metadata extraction)
   - `model/mlp_model.py`
4. **Retrieval & Fusion Engine:**
   - `distance_functions.py`
   - `fusion_methods.py`
   - `retrieval_engine.py`
   - `retrieval/distance.py`
   - `retrieval/evaluation.py`
5. **Phase 1, 2, and 4 Outputs & Manifests:**
   - `output/phase1/split_manifest.csv` (SHA256 verified)
   - `output/phase1/duplicate_audit.json`
   - `output/phase2/features.npz` (evaluated sample shapes and paths verified)
   - `output/phase2/feature_metadata.json` (normalization params verified)
   - `output/phase2/feature_extraction_config.json`
   - `output/phase4/rankings.npz`
   - `output/phase4/query_weights.csv`
   - `output/phase4/per_query_metrics.csv`
   - `output/phase4/per_category_metrics.csv`
   - `output/phase4/aggregate_metrics.json`
   - `output/phase4/retrieval_config.json`
   - `output/phase4/retrieval_validation_report.md`
6. **Execution Scripts & Test Suite:**
   - `scripts/run_phase4_retrieval.py`
   - `tests/test_phase3.py` (executed: 6 passed)
   - `tests/test_phase4.py`

---

## 9. Proposed Next Action & User Approval Gate

In accordance with the **Mandatory Phase Execution Policy**:
- **Do NOT begin Phase 5 automatically.**
- We present these audit findings to the user.
- Upon user approval, Phase 5 (statistical validation: Wilcoxon signed-rank tests, Holm-Bonferroni correction, stratified bootstrap confidence intervals, and category sensitivity analysis) will be executed using the verified saved Phase 4 per-query metrics into a dedicated `output/phase5/` directory without modifying or overwriting any Phase 1–4 artifacts.
