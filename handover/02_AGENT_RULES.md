# Agent Rules — Persistent Operating Instructions

1. Treat `handover/01_PROJECT_CONTEXT_MASTER.md` as the project context authority, except where it explicitly marks a value stale, unresolved, or reported-but-not-included.
2. Before every task, inspect the relevant source code, config, and saved outputs. Do not infer that a reported metric was reproduced merely because a report states it.
3. Work one phase or one approved task at a time. At the end, provide exact changed files, commands, test results, output paths, figure paths, and unresolved issues. Then STOP and wait for explicit user approval.
4. Never move to the next phase automatically.
5. Never overwrite prior phase outputs, figures, checkpoints, manifests, reports, or raw results. Create a new run directory with a timestamp or version suffix and append a row to `handover/OUTPUT_REGISTER.csv`.
6. Keep all outputs in the existing naming scheme: `output/phaseN/` and `figures/phaseN_*.png`. Add reproducibility metadata: seed, config, software versions, input hashes, timestamp, and command.
7. Preserve exact dataset split, query identities, category labels, and duplicate handling. Never silently regenerate splits.
8. Keep the held-out query set completely isolated from model training, validation, normalization fitting, hyperparameter tuning, and early stopping.
9. Do not change M1–M6 definitions or M6 fixed weights without explicit approval. Never confuse feature-level and distance-level fusion.
10. Do not invent triplet sampling, training details, or missing logs. If the methodology/code is ambiguous, stop and ask for approval.
11. Do not claim statistical significance when adjusted tests are non-significant. Distinguish confidence intervals from hypothesis tests and single-dataset query-level inference from independent multi-dataset validation.
12. Do not claim cross-dataset generalisation from unresolved Oxford-IIIT Pet values.
13. Do not edit the manuscript until the exact current source and assets are identified. Keep a separate Springer LNNS build; do not mistake IEEEtran source for LNNS.
14. No screenshots of code in the paper. Use publication-quality figures with captions and readable labels.
15. All manuscript metrics must match saved CSV/JSON/NPZ outputs. Maintain a claim-to-artifact traceability table.
16. If a tool or dependency is unavailable, report the limitation; do not pretend it ran.
17. Never use external web research unless the user asks for it or it is necessary for current publisher/template guidance. Clearly distinguish external sources from project evidence.
