# Reproducibility Checklist: ICAIN 2026 Paper ID 1567

This checklist conforms to standard machine learning reproducibility guidelines (NeurIPS / CVPR / Springer):

### 1. Dataset & Partitions
- [x] Exact dataset source and version specified (Wang Corel-1K, 1,000 images, 10 categories).
- [x] Deterministic random seed documented (`seed=42`).
- [x] Query and gallery splits explicitly documented (`output/phase1/split_manifest.csv`).
- [x] Exact counts disclosed (899 gallery, 100 queries, 700 internal training, 199 validation).
- [x] Duplicate handling verified (near-duplicate `horses/761.jpg` excluded from gallery).
- [x] Held-out query isolation strictly enforced across all normalization and training stages.

### 2. Method & Model Architecture
- [x] Complete feature mathematical definitions documented (9-D HSV moments, 4-D GLCM properties).
- [x] Normalization fitting scope disclosed (gallery-only Min-Max bounds).
- [x] MLP layer dimensions, activation functions, and output Softmax constraint documented (`13-16-8-2`).
- [x] Triplet loss margin ($m = 0.15$), learning rate ($10^{-3}$), and Adam optimizer documented.
- [x] Model checkpoint preserved and verified (`model/adaptive_weight_mlp.pt`).

### 3. Evaluation & Metrics
- [x] Evaluation definitions mathematically specified (Precision@10, Recall@10, F1@10, full-gallery mAP).
- [x] F1@10 calculated at query-level using unrounded precision and recall prior to averaging.
- [x] Paired statistical hypothesis tests executed (two-sided Wilcoxon signed-rank test).
- [x] Multiple testing correction applied (Holm-Bonferroni step-down procedure).
- [x] Effect sizes ($d_z$) and bootstrap confidence intervals reported.

### 4. Computational & Baseline Reproducibility
- [x] Baseline implementations fully disclosed (D1: LBP, D2: ResNet-50, D3: EfficientNet-B0, D5: CLIP ViT-B/32).
- [x] Model checkpoints, layers, resolutions, and normalizations recorded in `feature_inventory.json`.
- [x] Execution hardware disclosed (Intel Core i7-14650HX, 16 GB DDR5, Windows 11).
- [x] Standalone component latencies, throughputs, and memory storage measured.
- [x] External transfer on official Oxford-IIIT Pet dataset executed and reconciled.
- [x] Complete environment specification provided (`requirements.txt`, `environment.yml`).
- [x] Automated unit test suite with 53 passing tests verifying numerical integrity.
