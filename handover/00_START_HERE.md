# START HERE — ICAIN 2026 CBIR Project Master Handover

**Project:** ICAIN 2026, Springer LNNS  
**Paper ID:** 1567  
**Title:** Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features  
**Current Code Snapshot:** Phases 1 through 10 fully completed, audited, and verified.

---

## 🚨 MANDATORY FIRST ACTION FOR THE ANTIGRAVITY AGENT

1. **Read `ANTIGRAVITY_AGENT_ONBOARDING_INSTRUCTIONS.md` in the project root immediately.**
   It contains your complete onboarding directive, system walkthrough instructions, mathematical formulation, frozen benchmark values, and research integrity rules.
2. **Execute the verification walkthrough:**
   - Run `pytest tests/` (all 58 automated unit tests must pass).
   - Inspect the 7 publication-ready tables in `manuscript_tables/`.
   - Inspect the 9 publication-quality figures (300 DPI) in `publication_figures/`.
   - Inspect the technical audit reports: `final_research_audit.md`, `reproducibility_checklist.md`, and `final_results_summary.md`.
3. **DO NOT overwrite, retrain, or regenerate existing frozen outputs.** All experimental results under `output/phase1/` through `output/phase10/` are immutable and cryptographically audited in `experiment_manifest.json`.
4. **Current Workflow Focus:** Assist the user with the final manuscript writeup and revision following `manuscript_integration_recommendation.md`.

---

## Project State at Handover (Phases 1 to 10 Complete)

- **Phase 1:** Dataset audit & corrected split (899 gallery, 100 queries, `horses/761.jpg` purged duplicate).
- **Phase 2:** 13-D HSV + GLCM descriptor extraction and gallery-only Min-Max normalization.
- **Phase 3:** M1–M4 fixed baselines implemented and verified.
- **Phase 4:** MLP training provenance audited and verified (`adaptive_weight_mlp.pt`, 378 parameters).
- **Phase 5:** Corrected Corel-1K benchmark evaluation (M1–M6, M5 mAP = 0.4798, P@10 = 0.7100).
- **Phase 6:** Inferential statistical validation (Wilcoxon tests, sign-flip permutations, bootstrap CIs, Holm correction).
- **Phase 7:** Gallery-composition sensitivity & Leave-One-Category-Out (LOCO) analysis.
- **Phase 8:** Handcrafted and deep-learning baselines comparison (D1 LBP, D2 ResNet-50, D3 EfficientNet, D5 CLIP).
- **Phase 9:** External zero-shot transfer on Oxford-IIIT Pet (37 breeds) & computational scalability profiling.
- **Phase 10:** Final research audit, 7 manuscript tables (LaTeX + CSV), 9 publication figures (300 DPI), and master manifests.

All 58 unit tests passing via `pytest tests/`.
Refer to `ANTIGRAVITY_AGENT_ONBOARDING_INSTRUCTIONS.md` for full instructions.
