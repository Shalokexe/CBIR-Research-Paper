# Master Project Context — ICAIN 2026 CBIR

## 1. Project identity and objective

- Conference: ICAIN 2026
- Paper ID: 1567
- Target publisher/style: Springer Lecture Notes in Networks and Systems (LNNS)
- Manuscript title: **Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features**
- Research question: whether query-adaptive fusion of handcrafted colour and texture descriptors improves CBIR over single-modality and fixed-fusion baselines.
- Implementation environment: Python project intended to be developed/run in Google Antigravity. The current handover ZIP is a project snapshot, not a live MCP server.

## 2. Core method — authoritative design

### 2.1 Descriptor
Each image has a 13-dimensional descriptor:
- Colour: 9 HSV statistics = mean, standard deviation, skewness for H, S, and V.
- Texture: 4 GLCM properties = contrast, correlation, energy, homogeneity; averaged across 0°, 45°, 90°, 135° orientations.

Feature extraction details implemented in this repository:
- RGB image converted to HSV; channel-wise moments.
- Population standard deviation.
- Skewness is third central moment divided by SD cubed; if SD < 1e-6, skewness is set to zero.
- GLCM: grayscale uint8, 256 levels, distance 1, four orientations, symmetric=True, normed=True.
- MinMax normalization fitted on the gallery only, then applied unchanged to query features.

### 2.2 MLP
- Input dimension: 13.
- Hidden layers: 16 and 8 units.
- LeakyReLU negative slope: 0.1.
- Output: 2 logits followed by Softmax.
- Output weights: colour `w_C(q)` and texture `w_T(q)`, non-negative and sum to one.
- Intended metric-learning loss:
  `L = max(0, D_M5^2(q,p) - D_M5^2(q,n) + 0.15)`
  where p is same-class positive, n is different-class negative, and `D_M5^2` is the adaptive weighted squared distance.
- Intended optimizer: Adam, learning rate 0.001, batch size 64.
- Intended early stopping: validation loss, patience 10; retain best validation-loss checkpoint.
- **Provenance warning:** this ZIP contains `model/adaptive_weight_mlp.pt` and MLP inference code, but its inventory does not show a dedicated training script, explicit triplet sampler, epoch log, validation-loss history, or checkpoint metadata report. The architecture and intended training procedure are described in `docs/extracted_methodology.txt`, but the actual training process is not independently reproducible from the visible source inventory. Audit checkpoint metadata and locate the original trainer/logs before asserting full training reproducibility. Do not invent a sampler or retrain without approval.

### 2.3 Retrieval distances and model definitions

Let `D_C(q,g)` be Euclidean distance over normalized 9D colour features and `D_T(q,g)` Euclidean distance over normalized 4D texture features.

- M1: colour-only, `D_C`.
- M2: texture-only, `D_T`.
- M3: equal feature-level concatenation `[0.5 C, 0.5 T]`, therefore squared-distance contribution is `0.25 D_C² + 0.25 D_T²`.
- M4: fixed feature-level concatenation `[0.7 C, 0.3 T]`, therefore squared-distance contribution is `0.49 D_C² + 0.09 D_T²`.
- M5: adaptive distance-level fusion:
  `D_M5²(q,g) = w_C(q) D_C²(q,g) + w_T(q) D_T²(q,g)`.
- M6: fixed distance-level control using the corrected reference mean query weights:
  `w_C = 0.578403890132904`, `w_T = 0.421596109867096`.
  This is a reference-query mean-weight control, not a validation-selected weight and not an ex-ante independent deployment weight. Do not replace it with the separate five-run sensitivity experiment's mean weights.

**Do not confuse feature-level and distance-level weighting.** M3/M4 feature coefficients are squared when expressed as squared Euclidean distance. M5/M6 distance-level coefficients are applied directly to squared modality distances.

## 3. Dataset and leakage protocol — corrected protocol

Corel-1K:
- 1,000 original images; 10 categories; 100 images per category.
- Near-duplicate pair: `training_set/horses/761.jpg` and `training_set/horses/776.jpg`, pHash Hamming distance 4.
- Exclude `761.jpg` from gallery; retain `776.jpg` as held-out query.
- Corrected evaluated set: 899 gallery + 100 held-out queries = 999.
- MLP train: 700 gallery images.
- MLP validation: 199 gallery images.
- Query: 100 images, 10/category.
- Training: 70/category; validation 20/category except horses 19 because of duplicate exclusion.
- Seed 42.
- Query set must remain isolated from training, validation, hyperparameter selection, early stopping, and normalization fitting.
- Strict group-disjoint split; no excluded duplicate in training/validation.

**Stale-document warning:** `docs/extracted_methodology.txt` in this ZIP still describes 900 gallery, 200 validation, and 900-gallery normalization in places. That is the older, uncorrected protocol. The actual corrected implementation/output snapshot uses 899 gallery, 100 queries, 700 train, 199 validation. Do not copy the stale counts into the paper. Update the methodology document only after comparing it against `config.py`, `output/phase1`, `output/phase2`, and split manifest.

## 4. Completed implementation phases and verified results

### Phase 1 — dataset audit and partitioning
- Source files: `data/dataset_loader.py`, `data/duplicate_audit.py`, `data/partitioning.py`, `scripts/run_phase1_audit.py`.
- Outputs: `output/phase1/duplicate_audit.json`, `output/phase1/split_manifest.csv`.
- Corrected counts: 899 gallery, 100 queries, 700 train, 199 validation.
- Exact SHA256 duplicates: 0.
- One near-duplicate pair identified: horses 761/776; 761 excluded, 776 retained as query.
- Leakage checks and deterministic partition tests passed in prior phase review.
- Phase 1 tests: 6 passed.

### Phase 2 — feature extraction
- Source files: `features/feature_extractor.py`, `features/normalization.py`, `scripts/run_phase2_extraction.py`.
- Outputs: `output/phase2/features.npz`, `features.csv`, config/metadata JSON, validation report.
- 999 evaluated images = 899 gallery + 100 queries.
- Raw and normalized feature parity with reference: max absolute difference 0.0; path parity true.
- Extraction approximately 46.8 seconds, 46.85 ms/image.
- Phase 2 tests: 6 passed; cumulative 12.

### Phase 3 — baseline retrieval
- Source files: `retrieval/distance.py`, `retrieval/evaluation.py`, `scripts/run_phase3_baselines.py`.
- Outputs: `output/phase3/rankings.npz`, per-query metrics (400 rows), per-category metrics (40 rows), aggregate metrics, report.
- M1–M4 exact reference mAP parity and ranking-index parity across all 89,900 query-gallery pairs per model.
- Phase 3 tests: 18/18 passed (cumulative suite at that checkpoint).

### Phase 4 — MLP inference and full M1–M6 retrieval
- Source files include `model/mlp_model.py`, `fusion_methods.py`, `distance_functions.py`, `retrieval_engine.py`, `scripts/run_phase4_retrieval.py`, `tests/test_phase4.py`.
- Model checkpoint: `model/adaptive_weight_mlp.pt`.
- Outputs: `output/phase4/rankings.npz`, `query_weights.csv`, `per_query_metrics.csv` (600 rows), `per_category_metrics.csv` (60 rows), aggregate metrics, retrieval config, validation report.
- Figures: `phase4_m1_m6_performance_comparison.png`, `phase4_m5_weight_distribution.png`, `phase4_category_weights.png`.
- Corrected protocol in the Phase 4 retrieval report: 899 gallery, 100 queries.
- M5 predicted weight means: colour 0.5784038901, texture 0.4215961099; SD each 0.1568; colour range 0.0956–0.7523; texture range 0.2477–0.9044.
- M6 uses those mean M5 query weights.
- Aggregate results:

| Model | P@10 | R@10 | F1@10 | mAP |
|---|---:|---:|---:|---:|
| M1 | 0.6190 | 0.068890 | 0.123982 | 0.404462 |
| M2 | 0.4680 | 0.052067 | 0.093709 | 0.350816 |
| M3 | 0.6990 | 0.077779 | 0.139982 | 0.465036 |
| M4 | 0.6730 | 0.074893 | 0.134786 | 0.433659 |
| M5 | 0.7100 | 0.079000 | 0.142180 | 0.479794 |
| M6 | 0.6960 | 0.077443 | 0.139378 | 0.460536 |

M5 has the highest aggregate values among these six configurations in this evaluation. This is a descriptive result only; it does not establish statistical significance or broad generalisation. Category-wise winners vary: M3, M4, M5, and M6 each lead in some categories. Do not say M5 wins every category.

## 5. Later statistical validation already performed in the associated research workflow

The following results were produced in a subsequent saved-results-only statistical audit in the associated ChatGPT workflow. They are recorded here for context, but the raw Phase 5 outputs are not included in this source ZIP; keep them as reported values until the corresponding CSV/ZIP artifacts are copied into the project and verified.

- M5−M3 mean paired AP difference: +0.014758; Wilcoxon two-sided p=0.3259; Holm-adjusted p=0.6519.
- M5−M6 mean paired AP difference: +0.019258; Wilcoxon two-sided p=0.5543; Holm-adjusted p=0.6519.
- Stratified paired-bootstrap 95% CI: M5−M3 [0.007476, 0.023216]; M5−M6 [0.011969, 0.028012].
- Neither primary paired Wilcoxon comparison is statistically significant after Holm adjustment.
- Flowers is a strong category-specific driver: contribution to category-mean differences is +0.167349 for M5−M3 and +0.210336 for M5−M6. Excluding Flowers, the mean differences are −0.002196 and −0.001973, respectively.
- Bootstrap CI and Wilcoxon test answer different inferential questions; do not interpret a positive bootstrap interval as making the Wilcoxon comparison significant.
- 100 queries from one dataset do not constitute independent multi-dataset validation.

## 6. Five-run sensitivity analysis — separate, older protocol; do not mix

A separate five-seed/split experiment used an older 900-gallery/200-validation protocol and overlapping query identities (399 unique identities across 500 query slots). It is descriptive seed-and-split sensitivity, not independent replication and not five fixed-split training seeds.

- M5 mAP mean ± sample SD: 0.4858 ± 0.0114.
- M6 mAP mean ± sample SD: 0.4682 ± 0.0128.
- Paired mAP difference: +0.017592, SD 0.004008.
- M6 here was a mean of M5 weights over the 100 held-out query feature vectors, not validation-selected or ex-ante independent.
- These numbers are not from the corrected fixed protocol. Do not combine them with the corrected single-run M1–M6 table or use them as inferential evidence of independent replication.

## 7. Oxford-IIIT Pet cross-dataset transfer — unresolved; do not publish numerical claims

Earlier frozen transfer artifacts were audited with ranking verification, but the latest manuscript PDF reported different Pet mAP values. The current source ZIP does not contain the transfer artifacts needed to reconcile the discrepancy. Earlier audited values were M5 0.033882, M6 0.033683, M7 0.033829; a later PDF stated M5 0.033832 and M6 0.033715. These conflict. Until the exact frozen rankings and metrics are supplied and reconciled, remove Pet numerical claims or describe cross-dataset transfer as a limitation/future work. Do not claim universal generalisation.

## 8. Manuscript and reviewer issues already identified

Reviewer concerns:
1. Clarify novelty relative to query-adaptive CBIR.
2. Fully specify MLP architecture, inputs, training, loss, and query-specific weights.
3. Compare deep-learning CBIR and handcrafted baselines.
4. Report statistical significance and confidence intervals on independent queries/datasets, not only repeated runs.
5. Explain duplicate removal, relevance labels, and gallery composition; test sensitivity to gallery composition.
6. Assess Oxford-IIIT Pet transfer, scalability, computational cost, and generalisability.
7. Add explicit bullet contribution list, high-resolution figures, baseline comparison, complete references, evaluation parameter table; remove code screenshots.

Manuscript audit issues:
- The exact source used for the latest PDF was not identified.
- Older source `CBIR_paper_figures_placed.tex` used IEEEtran, not Springer LNNS.
- Latest PDF abstract count wording is incorrect: six total configurations means five alternatives, not “six other” configurations.
- Older LaTeX has stale methodology/objective and obsolete seed/M6 claims.
- Corrected Corel results must replace older numbers where applicable.
- Do not claim statistical significance for M5 improvements.
- Use official Springer LNNS template and verify exact source before editing.
- Keep paper ID 1567 and title consistent.

## 9. Next actions — ordered, gated workflow

### Immediate: Phase 4 provenance audit (do this first)
1. Inspect `model/adaptive_weight_mlp.pt` metadata/state and `config.py`.
2. Search the complete repository for any training script, triplet sampler, saved training history, checkpoint epoch, validation loss, random seed, and checkpoint-selection metadata.
3. Compare actual training/validation split against the corrected 700/199 split manifest.
4. Determine whether triplet loss was computed on the intended M5 adaptive weighted squared distance.
5. Verify whether training was deterministic and whether the 100 query images were excluded from all training/validation/tuning.
6. Verify M6 values exactly match the reference constants and that M6 is the reference mean-query-weight control.
7. Produce `output/phase4/phase4_provenance_audit.md` and add audit status to `handover/OUTPUT_REGISTER.csv`.
8. Stop and show the user findings. Do not retrain or modify results without approval.

### After user approval: Phase 5
- Re-run/verify paired per-query statistical comparisons using saved Phase 4 AP values.
- Include M5 vs M3 and M5 vs M6 primary comparisons; Wilcoxon signed-rank, Holm correction, paired/stratified bootstrap CIs, category sensitivity, per-query wins/losses.
- Clearly state one-dataset limitation and avoid overclaiming significance.
- Save all raw tables, code, config, report, and figures under a new `output/phase5/` directory; never overwrite Phase 4.

### After user approval: Phase 6
- Publication-quality figures and complete evidence inventory.
- Manuscript correction using exact current `.tex`, `.bib`, figures, and Springer LNNS template.
- Reconcile all tables against machine-readable outputs.
- Resolve or remove unsupported Oxford-IIIT Pet numerical claims.
- Final reproducibility, references, formatting, and compile checks.

## 10. Reporting discipline

Every result in the paper must trace to a saved artifact and run configuration. Label results as:
- **Verified in this ZIP** — directly present and inspectable.
- **Reported from later workflow** — known result but source artifact not in this ZIP.
- **Unresolved** — do not publish as fact.
- **Proposed** — not yet executed.

Never fabricate missing outputs, logs, citations, dataset evidence, or statistical significance.
