# Project Specification

## 1. Final project goal

Build a research-grade Urdu misinformation detection **and analysis** platform whose central contribution is auditing whether Urdu misinformation-detection models learn genuine linguistic signal or dataset-specific shortcuts (article length, source style) — combining rigorous cross-dataset evaluation, shortcut/bias analysis, and explainability with professional software engineering and a responsibly-framed public deployment. Full research grounding: `RESEARCH_VALIDATION_REPORT.md` (Phase 0/1) and `MASTER_PROJECT_BLUEPRINT.md` (Phase 2).

## 2. Final project title

**"Beyond In-Domain Accuracy: Auditing Shortcut Learning and Cross-Dataset Generalization in Urdu Misinformation Detection"** (formal/thesis title). Short/GitHub-facing name: **"Urdu Misinformation Detection: Cross-Dataset Generalization & Shortcut Analysis."**

## 3. Research question

> Do Urdu misinformation detection models learn genuine linguistic signals related to misinformation, or do they exploit dataset-specific shortcuts such as article length, source characteristics, domain characteristics, or other artifacts?

Five research questions (RQ1–RQ5), each with hypothesis/dataset/variables/metrics/expected-outcome criteria, are fully specified in `MASTER_PROJECT_BLUEPRINT.md` Part 4 and operationalized in `EXPERIMENT_PLAN.md`.

## 4. What this project deliberately is not

Not a claim to beat the current published SOTA (96.2% accuracy, Feb 2026). Not a new-dataset contribution. Not a fact-verification or ground-truth-determination system. Not a large, feature-maximalist product — every feature must trace to either the research or the MUST/SHOULD-HAVE product scope in `MASTER_PROJECT_BLUEPRINT.md` Part 17.

## 5. Document index

| Document | Covers |
|---|---|
| `RESEARCH_VALIDATION_REPORT.md` | Phase 0/1 — problem validation, literature review, dataset landscape |
| `MASTER_PROJECT_BLUEPRINT.md` | Phase 2 — full research design (RQs, contributions, experiments, evaluation, explainability, responsible AI, thesis structure) |
| `DECISION_REGISTER.md` | Every final technical decision, why, and rejected alternatives — binding |
| `ARCHITECTURE.md` | System diagram, tech stack table, full repo tree |
| `FRONTEND_SPECIFICATION.md` | Pages, design system, component tree, user flows |
| `BACKEND_SPECIFICATION.md` | API contract, service layering, config |
| `ML_SPECIFICATION.md` | Inference pipeline internals, model loading, tokenization |
| `DATASET_PLAN.md` | Operationalized dataset acquisition/audit/split pipeline |
| `EXPERIMENT_PLAN.md` | Operationalized experiment matrix and execution order |
| `TESTING_STRATEGY.md` | Tooling and test plan per layer |
| `SECURITY.md` | SSRF prevention, input validation, rate limiting |
| `DEPLOYMENT_PLAN.md` | Vercel + HF Spaces deployment, env vars, CI/CD |
| `REPRODUCIBILITY.md` | Seeds, versions, config-driven experiments |
| `ROADMAP.md` | Milestone-by-milestone implementation plan for Claude Code |
| `THESIS_PLAN.md` | Artifact-to-thesis-chapter mapping |
| `RESEARCH_PAPER_PLAN.md` | How results convert into a paper draft |
| `GITHUB_PLAN.md` | Repository presentation strategy |
| `CLAUDE.md` | Entry point and binding rules for the implementing agent |

## 6. Responsibility split

**Claude Code will:** write and refactor all code (frontend, backend, ML pipeline, tests, CI config, deployment config); implement each `ROADMAP.md` milestone in order; run and fix tests/lint/type checks; maintain documentation synchronization per `CLAUDE.md` rule 8; generate the model card/dataset card from real evaluation artifacts (never hand-write numbers into them); flag `DECISION REQUIRED` items rather than guessing.

**Rehman will:** create accounts (Vercel, Hugging Face, GitHub, Weights & Biases) and provide any resulting tokens/credentials via `.env`, never committed; run the GPU training steps that need a hosted GPU — Colab or Kaggle (Milestones 3–5 in `ROADMAP.md` — Claude Code prepares the exact runnable scripts/notebooks, Rehman executes them and returns the resulting checkpoints/metrics); review all research results before they're written into `docs/model_card.md`, `docs/dataset_card.md`, or `thesis/`; approve any change to the research methodology (`DECISION_REGISTER.md` R-series) before it's implemented; verify real-world outputs of the deployed demo; make final academic judgment calls (what goes in the thesis, how findings are framed); personalize and write the actual thesis/SOP/application prose (this project's documents are inputs to that writing, not a substitute for it); understand and be able to defend every design decision at a viva, using `MASTER_PROJECT_BLUEPRINT.md` Part 32 as a study guide, not a script to memorize.

**Shared / handoff points:** every hosted-GPU training run is a handoff — Claude Code prepares the script/config, Rehman runs it on Colab or Kaggle (hosted GPU access is not available to the coding agent; `research/src/notebook_env.py` makes one notebook work on either) and commits the resulting checkpoint reference + metrics back into the repo for Claude Code to continue building against.

## 7. Definition of "done" for the whole project

All REQUIRED experiments (`EXPERIMENT_PLAN.md`) executed with real, non-fabricated results; the deployed demo live on Vercel + HF Spaces, passing its own acceptance tests; documentation complete and internally consistent (no doc contradicts another); thesis chapters 1–22, 25–27 (`THESIS_PLAN.md`) have real content, not placeholders; Rehman can answer every question in `MASTER_PROJECT_BLUEPRINT.md` Part 32 using this project's actual measured numbers.
