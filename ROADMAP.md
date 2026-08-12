# Implementation Roadmap

Milestones are sequential and gated — do not start milestone N+1 until milestone N's acceptance criteria are met (`CLAUDE.md` rule 12). Each milestone ends with a git commit at the stated point. This roadmap operationalizes `MASTER_PROJECT_BLUEPRINT.md` Part 36 into concrete, Claude-Code-executable steps referencing the specific spec files.

## Milestone 0 — Repository scaffold
**Objective:** create the skeleton every later milestone builds into.
**Prerequisites:** none.
**Files created:** the full directory structure from `ARCHITECTURE.md` Section 4 (empty `__init__.py`/`.gitkeep` placeholders where needed), `.gitignore`, `.env.example`, `LICENSE` (MIT), root `README.md` (stub, filled in properly at Milestone 10 per `GITHUB_PLAN.md`), `backend/pyproject.toml` + `requirements.txt` (pinned per `REPRODUCIBILITY.md`), `frontend/package.json` (Next.js + TS + Tailwind + shadcn init), `.github/workflows/backend-ci.yml` + `frontend-ci.yml` (`DEPLOYMENT_PLAN.md` Section 5), and **`docs/responsible_ai.md`** — created now, early, because `CLAUDE.md` rule 14 makes it the single source of truth every later milestone's disclaimer text must quote; its content is copied verbatim from `MASTER_PROJECT_BLUEPRINT.md` Part 16's "Standing disclaimer text" block, not paraphrased.
**Tasks:** scaffold both apps with their respective CLIs (`npx create-next-app`, a minimal FastAPI skeleton), wire the empty CI workflows so they at least run (even against near-empty test suites).
**Tests:** CI workflows execute successfully (green on an empty/trivial test).
**Acceptance criteria:** `git clone` → `npm install && npm run dev` (frontend) and `pip install -r requirements.txt && uvicorn app.main:app` (backend) both start without error, serving placeholder content.
**Commit point:** `chore: scaffold repository structure`.

## Milestone 1 — Dataset acquisition
**Objective:** raw data present and checksummed.
**Files:** `research/src/data/download.py`, `research/data/raw/MANIFEST.sha256`, `research/data/raw/{ax_to_grind,notri_fact}/`.
**Tasks:** implement `download.py` per `DATASET_PLAN.md` Section 2 step 1; attempt UFND link resolution (`DECISION_REGISTER.md` U1) and log the outcome regardless of success.
**Tests:** a data-validation test confirming file presence + checksum match.
**Acceptance criteria:** raw files present, checksums recorded, row counts approximately match `DATASET_PLAN.md`'s documented sizes; `DECISION_REGISTER.md` updated with the UFND outcome either way.
**Commit point:** `data: acquire Ax-to-Grind and Notri-Fact raw datasets`.

## Milestone 2 — Dataset audit and cleaning
**Objective:** the Dataset Quality Report + Risk Register exist as real artifacts, and the cross-dataset duplication gate is resolved.
**Files:** `research/src/data/{validate,clean,dedup,audit,split}.py`, `research/data/audit/*`, `research/data/splits/*`.
**Tasks:** implement per `DATASET_PLAN.md` Section 2 steps 2–6, in order. Run the cross-dataset near-duplicate scan and resolve any overlap before proceeding. Use the real subword-length distribution from `audit.py`'s output to finalize `ML_SPECIFICATION.md` Section 3's sequence-length `DECISION REQUIRED` (U2) — update that doc with the resolved number.
**Tests:** schema/data-validation tests (`TESTING_STRATEGY.md`); an explicit test asserting zero cross-dataset near-duplicates post-cleanup.
**Acceptance criteria:** **hard gate** — zero cross-dataset duplication confirmed; `research/data/audit/` contains the length/vocabulary/source/structure reports and figures; `ML_SPECIFICATION.md` U2 resolved.
**Commit point:** `data: audit, clean, deduplicate, and split datasets`.

## Milestone 3 — Classical baselines + length-only baseline
**Objective:** experiments A, B, H complete with real metrics.
**Files:** `research/src/models/classical.py`, `length_baseline.py`, `research/src/experiments/run_in_domain.py` (classical mode), `research/src/experiments/run_shortcut_analysis.py` (length-only mode).
**Tasks:** per `EXPERIMENT_PLAN.md` Section 2, steps 1–2.
**Tests:** unit tests for feature extraction determinism; metrics-output schema test.
**Acceptance criteria:** `research/results/metrics/A_*.json`, `B_*.json`, `H_*.json` exist with real numbers; sanity-checked against `MASTER_PROJECT_BLUEPRINT.md` RQ1/RQ3 expected-range language (not required to match, just sane — e.g., not below chance).
**Commit point:** `research: classical and length-only baselines (Experiments A, B, H)`.

## Milestone 4 — Transformer training (in-domain)
**Objective:** experiments C, D complete, 3 seeds each, on Ax-to-Grind.
**Files:** `research/src/models/transformer.py`, `research/configs/model_{mbert,xlmr}.yaml`.
**Tasks:** per `EXPERIMENT_PLAN.md` Section 2 step 3. **This milestone requires Colab GPU — Claude Code prepares the exact runnable script/notebook; Rehman executes it and commits back the resulting checkpoints (pushed to a staging HF Hub location) + metrics** (`PROJECT_SPECIFICATION.md` Section 6 handoff).
**Tests:** checkpoint-loading regression test (`test_ml_sanity.py`) once a checkpoint exists.
**Acceptance criteria:** `research/results/metrics/C_*.json`, `D_*.json` for all 3 seeds; seed variance reported; checkpoints referenced by HF Hub repo ID + revision.
**Commit point:** `research: transformer in-domain training (Experiments C, D)`.

## Milestone 5 — Cross-dataset, shortcut, mitigation, explainability, error analysis
**Objective:** experiments F, G, I, N, O complete (REQUIRED set finished).
**Files:** `run_cross_dataset.py`, `run_shortcut_analysis.py` (length-ablation mode), `research/src/explainability/integrated_gradients.py`, `research/src/evaluation/error_analysis.py`.
**Tasks:** per `EXPERIMENT_PLAN.md` Section 2 steps 4–7, strictly after Milestone 2's dedup gate is confirmed passed.
**Tests:** IG completeness-axiom sanity check; error-analysis sampling reproducibility test (fixed seed → same sample).
**Acceptance criteria:** all REQUIRED experiments (`MASTER_PROJECT_BLUEPRINT.md` Part 12) have real results; `research/results/error_samples/` populated; `research/results/figures/` has the RQ2/RQ3/RQ5 figures from `MASTER_PROJECT_BLUEPRINT.md` Part 31. **This is the point at which the project's central empirical finding exists and can be checked against the Haroon (2026) reference numbers.**
**Commit point:** `research: cross-dataset generalization, shortcut analysis, and explainability (REQUIRED set complete)`.

## Milestone 5.5 — Strongly recommended experiments (E, J, K, L, M)
**Objective:** the EXTENSIONS set (`MASTER_PROJECT_BLUEPRINT.md` Part 34), attempted only after Milestone 5 is fully green.
**Tasks:** per `EXPERIMENT_PLAN.md` Section 2 steps 8–11.
**Acceptance criteria:** mitigation result (L/M) determines the deployed-checkpoint decision in `EXPERIMENT_PLAN.md` Section 7.
**Commit point:** `research: mitigation and extended shortcut analysis (EXTENSIONS set)`.

## Milestone 6 — Backend implementation
**Objective:** all endpoints in `BACKEND_SPECIFICATION.md` implemented and tested against a real (or the best-available-so-far) checkpoint.
**Files:** the full `backend/app/` tree (`ARCHITECTURE.md`).
**Tasks:** implement in dependency order: `config.py` → `ml/model_loader.py` + `ml/preprocess.py` (importing `research/src/data/clean.py` — verify parity per `ML_SPECIFICATION.md` Section 9) → `ml/inference.py` → `services/` → `security/ssrf.py` + `rate_limit.py` → `schemas/` → `routers/` → `main.py`.
**Tests:** full `backend/tests/` suite per `TESTING_STRATEGY.md` Section 1.
**Acceptance criteria:** `pytest` green, `ruff`/`mypy` clean, all endpoints manually verified via the auto-generated OpenAPI docs (`/docs`) against `BACKEND_SPECIFICATION.md`'s examples.
**Commit point:** `feat(backend): implement API per BACKEND_SPECIFICATION.md`.

## Milestone 7 — Frontend implementation
**Objective:** all pages/components in `FRONTEND_SPECIFICATION.md` implemented.
**Files:** the full `frontend/` tree.
**Tasks:** design tokens (`styles/globals.css`) first, then `ui/` primitives (shadcn init), then `layout/`, then page-by-page: `/`, `/analyze` (composing `analyze/` + `results/` components), `/methodology`, `/dataset`, `/model`, `/research` (composing `research/` chart components against `research/results/` data), `/responsible-use`, `/about`, `not-found`.
**Tests:** `frontend/tests/unit/` per `TESTING_STRATEGY.md` Section 2.
**Acceptance criteria:** `npm run test` green, `eslint`/`tsc` clean, every page renders against a running local backend (Milestone 6), RTL rendering manually verified on `/analyze`.
**Commit point:** `feat(frontend): implement pages and components per FRONTEND_SPECIFICATION.md`.

## Milestone 8 — Integration and E2E testing
**Objective:** frontend + backend + model work together end-to-end.
**Tasks:** wire `NEXT_PUBLIC_API_BASE_URL` for local dev, implement the one required Playwright flow (`TESTING_STRATEGY.md` Section 3).
**Tests:** `npx playwright test` green locally.
**Acceptance criteria:** the full main user flow (`FRONTEND_SPECIFICATION.md` Section 4) works against a locally-running backend with the real deployed-candidate checkpoint.
**Commit point:** `test: end-to-end integration flow`.

## Milestone 9 — Deployment
**Objective:** live public demo.
**Tasks:** per `DEPLOYMENT_PLAN.md` in full — Vercel project setup, HF Space setup + Dockerfile, model pushed to HF Hub, CORS configured, `DECISION_REGISTER.md` U4 (quantization) resolved from measured latency/memory.
**Acceptance criteria:** public URLs live, `/api/v1/health` returns `200`, the main flow works against the deployed (not local) backend, latency targets from `BACKEND_SPECIFICATION.md` met or a documented exception logged.
**Commit point:** `deploy: initial production deployment`.

## Milestone 10 — Documentation
**Objective:** `docs/model_card.md`, `docs/dataset_card.md`, `docs/responsible_ai.md`, and the real `README.md` (`GITHUB_PLAN.md`) all reflect actual, final results — not placeholders.
**Tasks:** `research/scripts/export_model_card.py` generates the model card from `research/results/metrics/` (never hand-typed numbers, per `CLAUDE.md` rule 2).
**Acceptance criteria:** a fresh reader can reproduce the pipeline from `README.md` alone (`REPRODUCIBILITY.md` Section 8's test).
**Commit point:** `docs: finalize model card, dataset card, and README`.

## Milestone 11 (stretch) — SHOULD-HAVE / STRETCH product features
Only after Milestones 0–10 are solid. URL extraction UI polish, prediction history (session-local), public API docs page — per `MASTER_PROJECT_BLUEPRINT.md` Part 17's SHOULD/STRETCH lists. Each addition gets its own commit and, if it changes an API contract, a `BACKEND_SPECIFICATION.md` update in the same commit.

## Milestone 12 — Thesis/paper support
Not code — populating `thesis/` per `THESIS_PLAN.md`, using the real artifacts from Milestones 1–5.5. Claude Code can assist in assembling figures/tables into the thesis document structure but the analytical prose is Rehman's per `PROJECT_SPECIFICATION.md` Section 6.
