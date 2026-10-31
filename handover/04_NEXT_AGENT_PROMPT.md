# Copy-Paste Prompt for the New Antigravity Agent

You are taking over an existing research implementation. Do not behave as if this is a new project.

First read, in order:
- `handover/00_START_HERE.md`
- `handover/01_PROJECT_CONTEXT_MASTER.md`
- `handover/02_AGENT_RULES.md`
- `handover/03_PHASE_STATUS.md`
- `handover/OUTPUT_REGISTER.csv`

Then inspect the repository. Your first and only task is a Phase 4 provenance audit. Confirm what is actually present in code and artifacts. In particular, inspect the MLP checkpoint metadata and determine whether training code, triplet sampling, epoch/validation history, checkpoint-selection metadata, and seed are available. Compare the training/validation partition with the corrected 700/199 split and confirm that the 100 held-out queries were excluded from training and tuning. Verify M5 uses the intended adaptive weighted squared-distance triplet objective if evidence exists. Verify M6 is the fixed reference mean-query-weight control (0.578403890132904, 0.421596109867096), not a validation-selected weight.

Do not train, retrain, overwrite files, edit the paper, or begin Phase 5. If anything is missing or ambiguous, state that explicitly and ask the user how to proceed.

Save the audit as a new file `output/phase4/phase4_provenance_audit.md` only after inspecting evidence; do not overwrite existing files. Update the output register with the audit file. At the end, give a concise audit summary, list every file inspected/created, state tests run (or not run), and STOP for user approval.
