# Experiment Plan (Operational)

Full research design (hypotheses, expected outcomes, support/reject criteria): `MASTER_PROJECT_BLUEPRINT.md` Parts 4, 11–13. This document is the execution checklist Claude Code follows in `research/src/experiments/`.

## 1. Models (final, per `DECISION_REGISTER.md` R4)

TF-IDF+Logistic Regression, TF-IDF+SVM, mBERT, XLM-RoBERTa-base. No other model is added without a new `DECISION_REGISTER.md` row and a stated research justification (`CLAUDE.md` rule 16).

## 2. Execution order (REQUIRED set first, always — `MASTER_PROJECT_BLUEPRINT.md` Part 12)

| Order | Experiment IDs | Script | Blocks on |
|---|---|---|---|
| 1 | A, B (classical in-domain) | `run_in_domain.py --models classical` | `DATASET_PLAN.md` split gate |
| 2 | H (length-only baseline) | `run_shortcut_analysis.py --mode length-only` | Step 1 |
| 3 | C, D (mBERT, XLM-R in-domain, 3 seeds each) | `run_in_domain.py --models transformer` | Step 1 (pipeline validated on cheap models first) |
| 4 | F, G (cross-dataset zero-shot, all 4 models, both directions) | `run_cross_dataset.py` | **Dedup gate in `DATASET_PLAN.md` step 4 must show zero cross-dataset near-duplicates before this runs** |
| 5 | I (length-ablation, replicate the 50-word cap design) | `run_shortcut_analysis.py --mode length-ablation` | Step 3 |
| 6 | N (Integrated Gradients on a sampled subset) | `research/src/explainability/integrated_gradients.py` | Step 3 (needs a trained checkpoint) |
| 7 | O (error analysis categorization) | `research/src/evaluation/error_analysis.py` | Steps 3–4 |
| 8 | E (in-domain on Notri-Fact, all 4 models) | `run_in_domain.py --dataset notri_fact` | Step 1 (STRONGLY RECOMMENDED — do after REQUIRED set is fully green) |
| 9 | J (length-bucketed performance) | part of `run_shortcut_analysis.py --mode length-buckets` | Steps 3–4 |
| 10 | K (vocabulary/source shortcut scan) | folded into `DATASET_PLAN.md`'s `audit.py`, re-run against trained-model errors | Step 7 |
| 11 | L, M (mitigation: length-stratified retrain + re-test cross-dataset) | `run_mitigation.py` | Steps 4–5 |
| 12 | P (domain-adaptive pretraining) | STRETCH — only after 1–11 are complete with time to spare | — |

Steps 1–7 are the REQUIRED set (`MASTER_PROJECT_BLUEPRINT.md` Part 12) and must all be complete, with real results committed to `research/results/`, before frontend/backend implementation proceeds past a placeholder-checkpoint stage (`ROADMAP.md` gates this explicitly).

## 3. Config-driven execution

Every run reads a single `research/configs/experiment_matrix.yaml` entry mapping an experiment ID (A–P, matching `MASTER_PROJECT_BLUEPRINT.md` Part 11's table) to a model config (`configs/model_*.yaml`) and a data config (`configs/data.yaml`). `run_all.py` can execute the full REQUIRED set from this file in one command — no experiment is run from ad hoc, undocumented CLI flags (`REPRODUCIBILITY.md`).

## 4. Output contract

Every experiment writes: a JSON metrics file (`research/results/metrics/<experiment_id>_<model>_<seed>.json` — accuracy, macro-F1, weighted-F1, per-class precision/recall, confusion matrix) and, where applicable, a figure (`research/results/figures/`) matching the specific figure list in `MASTER_PROJECT_BLUEPRINT.md` Part 31. Nothing is considered "done" if it exists only as a notebook cell's transient output (`CLAUDE.md` rule 2).

## 5. Metric

Primary: Macro-F1 (`DECISION_REGISTER.md` R5). Full set per experiment: accuracy, macro-F1, weighted-F1, per-class precision/recall, confusion matrix; ROC-AUC/PR-AUC for in-domain runs only; calibration/ECE for the primary deployed checkpoint. 3 random seeds (`{42, 123, 2026}`) for every transformer run (C, D, and their F/G/L/M counterparts); classical baselines (A, B, H) run once (deterministic enough — `MASTER_PROJECT_BLUEPRINT.md` Part 13).

## 6. Explainability sample

A single, fixed, stratified sample of ~100–120 examples (seeded, `research/data/audit/error_analysis_sample.csv`) is used for **both** the error-analysis (O) and explainability (N) steps, so the two efforts examine the same cases rather than duplicating sampling effort (`MASTER_PROJECT_BLUEPRINT.md` Part 14).

## 7. Model selection for deployment

The checkpoint promoted to `backend`'s `MODEL_REPO_ID` (`BACKEND_SPECIFICATION.md`) is the **length-stratified, mitigated XLM-R checkpoint from experiment L**, not necessarily the highest raw in-domain-F1 checkpoint — this is a deliberate choice (`MASTER_PROJECT_BLUEPRINT.md` Part 20) to deploy the most defensible checkpoint to end users, consistent with the project's responsible-AI stance. If experiment L/M shows no meaningful improvement (a legitimate possible outcome per `EXPERIMENT_PLAN.md` Section 2 step 11's linked hypothesis in the blueprint), fall back to the unmitigated XLM-R checkpoint (D) and document that choice explicitly in `DECISION_REGISTER.md` and `docs/model_card.md`.
