# CLAUDE.md — Instructions for Claude Code

Read this file before making any change to this repository. It is the entry point into the full specification set in `docs/` (mirrored at repo root during the planning phase: `PROJECT_SPECIFICATION.md`, `ARCHITECTURE.md`, `FRONTEND_SPECIFICATION.md`, `BACKEND_SPECIFICATION.md`, `ML_SPECIFICATION.md`, `DATASET_PLAN.md`, `EXPERIMENT_PLAN.md`, `TESTING_STRATEGY.md`, `SECURITY.md`, `DEPLOYMENT_PLAN.md`, `REPRODUCIBILITY.md`, `ROADMAP.md`, `DECISION_REGISTER.md`, `THESIS_PLAN.md`, `RESEARCH_PAPER_PLAN.md`, `GITHUB_PLAN.md`, `MASTER_PROJECT_BLUEPRINT.md`, `RESEARCH_VALIDATION_REPORT.md`).

## What this project is

A research-grade Urdu misinformation detection **and analysis** platform. The research question: do Urdu misinformation classifiers learn genuine linguistic signal, or dataset-specific shortcuts (article length, source style)? Full context: `PROJECT_SPECIFICATION.md`. Do not lose sight of this while implementing — this is not "just" a classifier app.

## File precedence

`ARCHITECTURE.md` Section 4 is the single authoritative repository tree. `MASTER_PROJECT_BLUEPRINT.md` Part 21 contains an earlier, superseded draft tree (kept for its conceptual reasoning, not its literal paths) — if the two ever appear to disagree, `ARCHITECTURE.md` wins. `DECISION_REGISTER.md` is binding for every technology/architecture choice; where `MASTER_PROJECT_BLUEPRINT.md`'s Part 37 draft decisions differ from `DECISION_REGISTER.md`'s final ones (e.g., frontend framework, deployment split), `DECISION_REGISTER.md` wins.

## Before touching any file, read the relevant spec

| If you're working on... | Read first |
|---|---|
| Anything | `PROJECT_SPECIFICATION.md`, `DECISION_REGISTER.md` |
| Repository layout, tech stack, request flow | `ARCHITECTURE.md` |
| `frontend/` | `FRONTEND_SPECIFICATION.md` |
| `backend/` | `BACKEND_SPECIFICATION.md` |
| `research/` (data, training, experiments) | `DATASET_PLAN.md`, `EXPERIMENT_PLAN.md`, `ML_SPECIFICATION.md` |
| `backend/app/ml/`, `backend/app/services/inference_service.py`, `explainability_service.py` | `ML_SPECIFICATION.md` |
| Any test file | `TESTING_STRATEGY.md` |
| `backend/app/security/`, URL extraction | `SECURITY.md` |
| Deployment config, Dockerfile, CI workflows | `DEPLOYMENT_PLAN.md` |
| Anything that must be reproducible (configs, seeds, checkpoints) | `REPRODUCIBILITY.md` |
| What to build next, in what order | `ROADMAP.md` |
| Thesis chapter mapping, paper structure | `THESIS_PLAN.md`, `RESEARCH_PAPER_PLAN.md` |
| README, licensing, repo presentation | `GITHUB_PLAN.md` |

## Non-negotiable rules

1. **Read the relevant spec before modifying files.** Do not invent architecture, endpoints, folder locations, or naming conventions that aren't already documented — if you believe the spec is wrong, say so explicitly and propose a change to `DECISION_REGISTER.md` rather than silently deviating.
2. **Never fabricate results, metrics, benchmark numbers, or citations.** Every number in `research/results/`, every claim in `docs/model_card.md` or `docs/dataset_card.md`, and every citation in `thesis/` must trace to an actual executed experiment or a verified source. If a number isn't computed yet, leave it as `TBD` with a comment, never a plausible-looking placeholder.
3. **Never change research methodology (Part 4/11/13 of `MASTER_PROJECT_BLUEPRINT.md`, or `EXPERIMENT_PLAN.md`) without documenting the change and the reason in `DECISION_REGISTER.md` first.**
4. **Preprocessing must never drift between training and serving.** `backend/app/ml/preprocess.py` imports from `research/src/data/clean.py` — it does not reimplement cleaning logic. If you need to change text cleaning, change it in `research/src/data/clean.py` only.
5. **The two primary datasets (Ax-to-Grind, Notri-Fact) are never concatenated or merged for training.** This is a hard architectural constraint from the research design (`DECISION_REGISTER.md` R7), not a style preference.
6. **Run tests after every implementation step**, not just at the end of a milestone. `TESTING_STRATEGY.md` defines what "passing" means for each layer.
7. **Run lint/type checks** (`ruff`/`mypy` for Python, `eslint`/`tsc` for TypeScript) before considering any file complete.
8. **Keep documentation synchronized.** If an implementation detail diverges from a spec doc (e.g., an endpoint's response shape changes), update the spec doc in the same commit — specs must never silently go stale.
9. **Keep commits logically organized** — one milestone step per commit where practical, following the commit points listed in `ROADMAP.md`. Do not bundle unrelated changes.
10. **Never delete working functionality without justification** recorded in the commit message or `DECISION_REGISTER.md`.
11. **Prefer small, incremental changes** over large, hard-to-review diffs, especially in `research/` where correctness of data handling is safety-critical to the thesis's validity.
12. **Validate each milestone against its acceptance criteria (`ROADMAP.md`) before starting the next one.** Do not build the frontend against a research pipeline that hasn't passed its own validation gates (e.g., the cross-dataset duplication check in `DATASET_PLAN.md` is a go/no-go gate, not optional cleanup).
13. **Never invent a `DECISION REQUIRED` resolution silently.** Items marked `DECISION REQUIRED` in `DECISION_REGISTER.md` must be resolved via the stated resolution path and logged, or explicitly surfaced to Rehman — not guessed.
14. **Responsible-AI copy is centralized.** All disclaimer/confidence-language text lives in `docs/responsible_ai.md` and is imported (not retyped) wherever it's shown — in the frontend, the model card, and the README. Never write a new variant of "this proves the article is fake" or similar overclaiming language anywhere in the codebase.
15. **No secrets in code or git history.** Use `.env` / the platform's secret manager per `DEPLOYMENT_PLAN.md`; `.env.example` documents variable names only.
16. **Don't add a dependency, service, or piece of infrastructure "because it's common practice."** Every addition should trace to a stated need in a spec doc (`DECISION_REGISTER.md` explains why a database was rejected for the MVP, for example) — if you think something's missing, propose it via a new `DECISION_REGISTER.md` row rather than adding it ad hoc.

## Definition of done (per milestone)

A milestone (see `ROADMAP.md`) is complete only when: its stated acceptance criteria are met, its tests pass, lint/type checks pass, any touched spec doc is updated to match reality, and the corresponding artifacts (metrics/figures/checkpoints for research milestones; passing CI for engineering milestones) exist on disk — not only "the code looks like it should work."

## Local development quickstart (reference — full detail in `DEPLOYMENT_PLAN.md`)

```bash
# Backend
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend && npm install
npm run dev

# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm run test        # Vitest + RTL
npx playwright test                # E2E

# Research pipeline (Colab-oriented, see research/scripts/run_full_pipeline.sh)
cd research && pip install -r requirements.txt
python src/data/download.py
python src/data/audit.py
```

## Escalation

If something in the spec set is ambiguous, contradictory, or you believe genuinely wrong: stop, state the conflict clearly, and propose a resolution as a new row in `DECISION_REGISTER.md`. Do not proceed on a guess for anything that affects research validity (dataset handling, metrics, experiment design) — proceeding on a reasonable guess is acceptable only for cosmetic/implementation-detail decisions clearly marked as such in the relevant spec.
