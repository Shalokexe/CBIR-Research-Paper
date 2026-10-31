# START HERE — ICAIN 2026 CBIR Project Handover

**Project:** ICAIN 2026, Springer LNNS  
**Paper ID:** 1567  
**Title:** Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features  
**Snapshot supplied:** `research paper implement(1).zip`  
**Current code snapshot:** implementation through Phase 4 (M1–M6 retrieval and evaluation).

## Mandatory first action for the new Antigravity agent

1. Read `handover/PROJECT_CONTEXT_MASTER.md` completely.
2. Read `handover/AGENT_RULES.md` and `handover/PHASE_STATUS.md`.
3. Inspect the existing repository and compare it with the output register. Do not overwrite or regenerate existing outputs.
4. Start by auditing the Phase 4 training provenance gap described in the master context. The ZIP contains an MLP checkpoint and inference/evaluation code, but no evident MLP training script or training log. Do not claim training reproducibility until this is resolved.
5. Report the audit findings and proposed next action to the user. Do not begin a new phase, retrain, change methodology, or edit manuscript/results until the user approves.

## Project state at handover

- Phase 1: dataset audit and corrected split implemented.
- Phase 2: 13D HSV+GLCM feature extraction and gallery-fitted normalization implemented.
- Phase 3: M1–M4 baselines implemented and reference-parity checked.
- Phase 4: M1–M6 retrieval/evaluation outputs and MLP inference checkpoint are present.
- Phase 4 retrieval outputs report M5 mAP 0.479794 and M6 mAP 0.460536.
- **Important:** Phase 4's report is not proof that MLP training can be reproduced. Training code/logs/metadata need audit.
- Next intended workflow: finish Phase 4 provenance audit → user approval → Phase 5 statistical validation → user approval → Phase 6 publication figures/report and manuscript revision.

## Preserve the evidence

All existing artifacts under `output/phase1` through `output/phase4`, `figures/`, `model/`, and source files are immutable baseline evidence. If any rerun is approved, create a new timestamped run directory and update the output register; never silently overwrite the existing run.
