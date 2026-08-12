# Deployment Plan

Final split: **frontend → Vercel, backend → Hugging Face Spaces, model weights → Hugging Face Model Hub** (`DECISION_REGISTER.md` E12). All free-tier.

## 1. Frontend (Vercel)

Connect the GitHub repo's `frontend/` directory (Vercel supports monorepo subdirectory deploys natively — set the project root to `frontend/` in Vercel's dashboard). Auto-deploys on every push to `main` (preview deployments on PRs, a genuinely useful free feature for reviewing UI changes before merge). Environment variable: `NEXT_PUBLIC_API_BASE_URL` pointing at the deployed HF Spaces backend URL — the only env var the frontend needs (`DECISION_REGISTER.md` E16).

## 2. Backend (Hugging Face Spaces)

Space type: **Docker SDK** (not the Gradio/Streamlit SDK — this is a FastAPI service, not a Gradio app; a custom `backend/Dockerfile` is required). Tier: CPU basic (free). **Build context is the repository root, not `backend/`** (`DECISION_REGISTER.md` E18) — this is required because the Dockerfile must `COPY research/src/data/clean.py` in addition to `backend/app/`, to satisfy the train/serve preprocessing-parity rule (`ML_SPECIFICATION.md` Section 9). If HF Spaces' own build pipeline expects the Dockerfile at the Space repo root, mirror this by structuring the HF Space's git remote so its root corresponds to this project's repo root (i.e., push the whole monorepo to the Space, with the Space configured to use `backend/Dockerfile` as its Dockerfile path and the repo root as build context — HF Spaces supports a custom Dockerfile path via its README metadata `app_file`/`dockerfile` config). The Dockerfile: installs `backend/requirements.txt`, copies `backend/app/` and `research/src/data/clean.py`, exposes the port HF Spaces expects (`7860` by convention), runs `uvicorn app.main:app --host 0.0.0.0 --port 7860`. Model is **not** baked into the Docker image (would bloat build time/image size) — it's downloaded from the HF Model Hub at container startup via `from_pretrained`, which HF Spaces' own caching handles reasonably well across restarts.

Deploy mechanism: HF Spaces is itself a git repository — either (a) Claude Code maintains a `git remote` pointing at the Space and a documented manual `git push` step, or (b) a simple GitHub Actions step pushes `backend/` (via `git subtree push` or a sync action) to the Space's remote on merge to `main`. Start with (a) for simplicity; automate to (b) only once the manual flow is proven to work (avoid building CD machinery against an unproven deploy target).

CORS: `ALLOWED_ORIGINS` (`BACKEND_SPECIFICATION.md`) set to the exact Vercel production domain, plus `http://localhost:3000` for local development — never a wildcard `*`, since this is a real (if low-stakes) API with a defined single consumer.

## 3. Model (Hugging Face Model Hub)

Public model repo, `MODEL_REPO_ID` referenced in backend config (`BACKEND_SPECIFICATION.md`). Pushed via `research/scripts/export_model_card.py` (generates the card from real eval artifacts) + the standard `transformers`/`huggingface_hub` push flow, run by Rehman after each Colab training run that produces the checkpoint being promoted to "deployed" (`PROJECT_SPECIFICATION.md` Section 6 — this is one of the explicit handoff points).

## 4. Cold starts, memory, latency

HF Spaces free CPU tier has a cold-start delay after inactivity — acceptable for a portfolio demo, disclosed if it becomes a UX issue (e.g., a "waking up the model, this may take a moment" state on the frontend if health-check polling detects a cold start, a small addition to `FRONTEND_SPECIFICATION.md`'s flows if implemented). XLM-R-base fits comfortably in the free CPU tier's memory ceiling; if `DECISION_REGISTER.md` U4 (quantization) is resolved as "needed" after real measurement, `optimum`-based fp16/int8 quantization is the documented fallback — not attempted speculatively.

## 5. CI/CD (GitHub Actions, `.github/workflows/`)

`backend-ci.yml`: on PR to `main`, install `backend/requirements.txt`, run `pytest`, `ruff check`, `mypy`. `frontend-ci.yml`: on PR to `main`, `npm ci`, `npm run test` (Vitest), `npx playwright test`, `eslint`, `tsc --noEmit`. Neither workflow deploys anything — deployment is via each platform's native GitHub integration (Vercel) or the documented manual/semi-automated push (HF Spaces), per `DECISION_REGISTER.md` E15's reasoning against building custom CD machinery.

## 6. Environment variables (complete list, `.env.example` mirrors this with empty values)

| Variable | Where | Purpose |
|---|---|---|
| `MODEL_REPO_ID` | Backend | HF Hub model repo to load |
| `MODEL_REVISION` | Backend | Pinned revision (`REPRODUCIBILITY.md`) |
| `ALLOWED_ORIGINS` | Backend | CORS allow-list |
| `RATE_LIMIT_PER_MINUTE` | Backend | Rate limit config |
| `MAX_TEXT_LENGTH` / `MIN_TEXT_LENGTH` | Backend | Input bounds |
| `EXTRACTION_TIMEOUT_SECONDS` / `EXTRACTION_MAX_BYTES` | Backend | SSRF/extraction limits |
| `LOG_LEVEL` | Backend | Logging verbosity |
| `NEXT_PUBLIC_API_BASE_URL` | Frontend | Points at the deployed backend |

No variable in this list is a secret in the traditional sense (no API keys to third-party paid services are required anywhere in this architecture) — worth stating explicitly, since it's a direct consequence of the "no unnecessary infrastructure" decisions (E8, E10) elsewhere in `DECISION_REGISTER.md`.

## 7. Monitoring/logging

Vercel and HF Spaces both provide basic built-in request/error logs on their free tiers — sufficient for this project's scale; no separate observability stack (Sentry, Datadog, etc.) is justified and none is added.
