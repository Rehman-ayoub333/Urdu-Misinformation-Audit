# Architecture

Source of truth for system structure. Technology choices here are final per `DECISION_REGISTER.md` — do not substitute alternatives during implementation.

**Precedence note (resolves a contradiction found during the pre-coding audit):** `MASTER_PROJECT_BLUEPRINT.md` Part 21 contains an earlier draft repository tree (flat `src/`/`data/` at repo root) from before the frontend/backend technology stack was finalized. **Section 4 of this document is the authoritative, current repository tree** — it supersedes Part 21 entirely. Do not implement against Part 21's tree.

## 1. System overview

```
┌─────────────────────────────┐
│   Frontend (Next.js)        │
│   Deployed on Vercel        │
│                              │
│  Home / Analyze / Results /  │
│  Methodology / Dataset /     │
│  Model Info / Research /     │
│  Responsible Use / 404       │
└──────────────┬───────────────┘
               │ HTTPS (JSON, CORS-restricted to the Vercel origin)
               ▼
┌─────────────────────────────┐
│   Backend (FastAPI)          │
│   Deployed on HF Spaces      │
│   (Docker SDK, CPU basic)    │
│                              │
│  routers/ → services/ → ml/  │
└──────────────┬───────────────┘
               │ loads at container startup
               ▼
┌─────────────────────────────┐
│  Model weights + tokenizer   │
│  Hugging Face Model Hub      │
│  (public repo)               │
└───────────────────────────────┘

Offline (not part of the live request path):
┌─────────────────────────────┐
│  Research pipeline (ml/)     │
│  Run on Google Colab (T4)    │
│  data → audit → train →      │
│  evaluate → cross-dataset →  │
│  shortcut analysis →         │
│  mitigation → explainability │
│  → push checkpoint to HF Hub │
└───────────────────────────────┘
```

The **research pipeline** (dataset audit, training, evaluation, cross-dataset testing, shortcut analysis, explainability) and the **product** (frontend + backend serving one deployed checkpoint) are architecturally separate. The product only ever consumes a finished checkpoint from the Hub; it never trains or re-evaluates at request time. This separation is intentional: it means backend/frontend implementation (Milestones 6–9 in `ROADMAP.md`) can proceed in parallel with, or independently of, the research experiments (Milestones 2–5), as long as a placeholder/early checkpoint exists to develop against.

## 2. Technology stack (final)

| Layer | Choice | Decision ID |
|---|---|---|
| Frontend framework | Next.js 14+, App Router, TypeScript | E2 |
| Styling | Tailwind CSS + shadcn/ui + custom design tokens | E3 |
| Frontend state | React built-in state + typed fetch client | E4 |
| Charts | Recharts | E5 |
| Backend framework | FastAPI + Pydantic v2, Python 3.12 | E6, re-pinned by M4-2 |
| Backend layering | routers → services → ml | E7 |
| Database | None (MVP); SQLite if a stretch feature needs it | E8/E9 |
| Rate limiting | In-process IP-based (`slowapi`) | E10 |
| Article extraction | `trafilatura` | E11 |
| ML | PyTorch, Hugging Face `transformers` + `datasets`, `scikit-learn`, `captum` | — (`ML_SPECIFICATION.md`) |
| Deployment (frontend) | Vercel | E12 |
| Deployment (backend) | Hugging Face Spaces (Docker SDK) | E12 |
| Model hosting | Hugging Face Model Hub | E17 |
| CI | GitHub Actions | E15 |
| Backend testing | pytest + httpx TestClient | E13 |
| Frontend testing | Vitest + React Testing Library + Playwright (1 E2E flow) | E14 |
| Experiment tracking | Weights & Biases (free tier) or local JSON logging fallback | `MASTER_PROJECT_BLUEPRINT.md` Part 26 |

## 3. Request lifecycle (the one path that matters most)

```
User submits Urdu text or a URL on the Analyze page
  → POST /api/v1/analyze (frontend → backend)
  → routers/analyze.py validates the request shape (Pydantic)
  → if url: services/extraction_service.py fetches + extracts article text
      (SSRF checks per SECURITY.md happen here, before any text reaches the model)
  → services/inference_service.py:
      preprocess (ml/inference/preprocess.py, SAME function used in training)
      → tokenize (model-specific tokenizer, loaded once at startup)
      → forward pass → logits → softmax → label + confidence
  → response returned immediately (explanation is NOT computed yet — see below)
  → frontend renders the Results screen
  → user optionally requests the explanation
  → POST /api/v1/explain (separate call, referencing the analyze response's id)
  → services/explainability_service.py runs Integrated Gradients (ml/explainability/)
  → response returns highlighted spans
  → frontend renders the Explanation screen
```

Splitting `/analyze` and `/explain` into two calls (rather than always computing both) is a deliberate latency decision: Integrated Gradients is meaningfully more expensive than a forward pass, and most users care about the prediction first. See `BACKEND_SPECIFICATION.md` for the full contract and `ML_SPECIFICATION.md` for inference internals.

## 4. Complete repository structure

```
urdu-misinfo-audit/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── .env.example
│
├── PROJECT_SPECIFICATION.md, ARCHITECTURE.md, FRONTEND_SPECIFICATION.md,   # All 17 planning docs (this document set) live at
│   BACKEND_SPECIFICATION.md, ML_SPECIFICATION.md, DATASET_PLAN.md,          # the REPO ROOT — that is their real, permanent
│   EXPERIMENT_PLAN.md, TESTING_STRATEGY.md, SECURITY.md,                    # location (not docs/). Do not move or duplicate
│   DEPLOYMENT_PLAN.md, REPRODUCIBILITY.md, ROADMAP.md,                      # them into docs/ during Milestone 0 — that would
│   DECISION_REGISTER.md, THESIS_PLAN.md, RESEARCH_PAPER_PLAN.md,            # create exactly the two-copies-can-drift problem
│   GITHUB_PLAN.md, CLAUDE.md, MASTER_PROJECT_BLUEPRINT.md,                  # CLAUDE.md rule 8 exists to prevent.
│   RESEARCH_VALIDATION_REPORT.md
│
├── docs/                              # ONLY generated/authored docs (not the planning set above)
│   ├── model_card.md                  # generated by research/scripts/export_model_card.py once a checkpoint is deployed — never hand-written (CLAUDE.md rule 2)
│   ├── dataset_card.md                # generated from research/data/audit/ + DATASET_PLAN.md, once Milestone 2 completes
│   └── responsible_ai.md              # source of truth for all disclaimer copy — created at Milestone 0, content copied VERBATIM from MASTER_PROJECT_BLUEPRINT.md Part 16's "Standing disclaimer text" block; every other file that shows disclaimer text (frontend components, model card, README) imports/quotes this file, never retypes it
│
├── research/                          # Everything in MASTER_PROJECT_BLUEPRINT.md Parts 6-15, operationalized
│   ├── data/
│   │   ├── raw/                       # gitignored EXCEPT MANIFEST.sha256, which is committed
│   │   │                              # (REPRODUCIBILITY.md S3 makes it the dataset version id)
│   │   ├── clean/                     # gitignored
│   │   ├── processed/                 # gitignored
│   │   ├── splits/                    # committed (row-ID index files only)
│   │   └── audit/                     # committed (Dataset Quality Report + Risk Register outputs)
│   ├── src/
│   │   ├── data/
│   │   │   ├── download.py
│   │   │   ├── validate.py
│   │   │   ├── clean.py               # imported by backend/app/ml/preprocess.py — single source of truth
│   │   │   ├── dedup.py
│   │   │   ├── audit.py
│   │   │   └── split.py
│   │   ├── models/
│   │   │   ├── classical.py
│   │   │   ├── transformer.py
│   │   │   └── length_baseline.py
│   │   ├── experiments/
│   │   │   ├── run_in_domain.py
│   │   │   ├── run_cross_dataset.py
│   │   │   ├── run_shortcut_analysis.py
│   │   │   ├── run_mitigation.py
│   │   │   └── run_all.py
│   │   ├── evaluation/
│   │   │   ├── metrics.py
│   │   │   └── error_analysis.py
│   │   └── explainability/
│   │       └── integrated_gradients.py
│   ├── configs/
│   │   ├── data.yaml
│   │   ├── model_xlmr.yaml
│   │   ├── model_mbert.yaml
│   │   ├── model_classical.yaml
│   │   └── experiment_matrix.yaml
│   ├── notebooks/                     # exploration only, per REPRODUCIBILITY.md
│   │   ├── 01_dataset_exploration.ipynb
│   │   ├── 02_audit_walkthrough.ipynb
│   │   └── 03_explainability_examples.ipynb
│   ├── scripts/
│   │   ├── download_data.py
│   │   ├── run_full_pipeline.sh
│   │   └── export_model_card.py
│   ├── tests/                         # data-validation tests (TESTING_STRATEGY.md S1's
│   │   └── test_raw_data_integrity.py # "Data validation tests" — own pytest target
│   ├── requirements.txt               # pinned per REPRODUCIBILITY.md Section 1
│   ├── pyproject.toml                 # pytest config; pins Python 3.12
│   └── results/
│       ├── metrics/                   # committed (small, core evidence)
│       ├── figures/                   # committed
│       └── error_samples/             # committed
│
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app factory, CORS config, startup model load
│   │   ├── config.py                  # pydantic-settings
│   │   ├── routers/
│   │   │   ├── analyze.py
│   │   │   ├── explain.py
│   │   │   ├── meta.py                # /health, /model-info, /examples
│   │   ├── services/
│   │   │   ├── inference_service.py
│   │   │   ├── explainability_service.py
│   │   │   └── extraction_service.py
│   │   ├── ml/
│   │   │   ├── preprocess.py          # imports research/src/data/clean.py — no duplicated logic
│   │   │   ├── model_loader.py        # singleton load at startup
│   │   │   └── inference.py
│   │   ├── schemas/
│   │   │   ├── analyze.py
│   │   │   ├── explain.py
│   │   │   └── meta.py
│   │   ├── security/
│   │   │   ├── ssrf.py                # URL/IP validation, per SECURITY.md
│   │   │   └── rate_limit.py
│   │   ├── data/
│   │   │   └── examples.json          # curated example articles (static fixture, E8)
│   │   └── logging_config.py
│   ├── tests/
│   │   ├── test_analyze.py
│   │   ├── test_explain.py
│   │   ├── test_meta.py
│   │   ├── test_ssrf.py
│   │   ├── test_ml_sanity.py
│   │   └── conftest.py
│   ├── Dockerfile                     # HF Spaces deployment target — build context is the REPO ROOT, not backend/ (see DEPLOYMENT_PLAN.md and DECISION_REGISTER.md E18); it COPYs both backend/app/ and research/src/data/clean.py
│   ├── requirements.txt
│   └── pyproject.toml
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                   # Home/Landing
│   │   ├── analyze/page.tsx
│   │   ├── methodology/page.tsx
│   │   ├── dataset/page.tsx
│   │   ├── model/page.tsx
│   │   ├── research/page.tsx
│   │   ├── responsible-use/page.tsx
│   │   ├── about/page.tsx
│   │   └── not-found.tsx
│   ├── components/
│   │   ├── layout/ (Navbar, Footer, DisclaimerBanner)
│   │   ├── analyze/ (ArticleInput, UrlInput, ExampleArticlePicker, AnalyzeButton)
│   │   ├── results/ (PredictionCard, ConfidenceGauge, ExplanationView, HighlightedText)
│   │   ├── research/ (BenchmarkTable, CrossDatasetHeatmap, ConfusionMatrix, LengthDistributionChart)
│   │   └── ui/                        # shadcn/ui primitives
│   ├── lib/
│   │   ├── api-client.ts              # typed fetch wrapper matching BACKEND_SPECIFICATION.md exactly
│   │   ├── types.ts                   # mirrors backend Pydantic schemas
│   │   ├── constants.ts               # disclaimer text, sourced from docs/responsible_ai.md
│   │   ├── validation.ts              # client-side length/URL-shape checks (UX only — server re-validates, per SECURITY.md; never the trust boundary)
│   │   └── hooks/
│   │       ├── useAnalyze.ts          # encapsulates the analyze request lifecycle (idle/loading/success/error) consumed by AnalyzeForm
│   │       └── useExplain.ts          # encapsulates the explain request lifecycle consumed by ExplainButton/ExplanationView
│   ├── styles/
│   │   └── globals.css                # design tokens per FRONTEND_SPECIFICATION.md
│   ├── tests/
│   │   ├── unit/ (Vitest + RTL)     # plus setup.ts, registering jest-dom matchers
│   │   └── e2e/ (Playwright)
│   ├── public/
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── postcss.config.js              # Tailwind + autoprefixer pipeline
│   ├── tsconfig.json
│   ├── eslint.config.mjs              # flat config; eslint-config-next 16 exports flat arrays directly, so no FlatCompat shim
│   ├── vitest.config.ts               # E14 unit suite; excludes tests/e2e/
│   ├── playwright.config.ts           # E14 E2E; Chromium only, self-starting webServer
│   ├── components.json                # shadcn/ui CLI config (E3) — lets `npx shadcn add` emit working components at Milestone 7 with no reconfiguration
│   ├── package.json
│   └── package-lock.json              # committed, per REPRODUCIBILITY.md Section 1
│
├── thesis/                            # LaTeX/Markdown source, per THESIS_PLAN.md
│
└── .github/
    └── workflows/
        ├── backend-ci.yml
        └── frontend-ci.yml
```

This tree is authoritative. If Claude Code needs a file not listed here, it must be added to this tree (in a commit that touches `ARCHITECTURE.md`) before being created elsewhere — no orphan files outside this structure.
