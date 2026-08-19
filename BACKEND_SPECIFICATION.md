# Backend Specification

Stack: FastAPI + Pydantic v2, Python 3.12 (re-pinned from 3.11 by `DECISION_REGISTER.md` M4-2), layered `routers → services → ml` (`DECISION_REGISTER.md` E6/E7). No database (E8). This document is the binding API contract — response shapes here must match `frontend/lib/types.ts` exactly; if either side needs to change, update this document in the same commit (`CLAUDE.md` rule 8).

## 1. Layering

`routers/` — HTTP concerns only: request parsing (via Pydantic schemas in `schemas/`), calling the appropriate service, mapping service exceptions to HTTP status codes. No business logic lives here. (Terminology note: FastAPI's own docs and community call these "path operation functions," not "controllers" — they are the controller-equivalent layer in this architecture; use "router" consistently in code/comments to match `ARCHITECTURE.md`'s tree, not "controller.")

`services/` — business logic: `InferenceService` (prediction), `ExplainabilityService` (Integrated Gradients), `ExtractionService` (URL → text, with SSRF checks per `SECURITY.md`). Services raise typed exceptions (`ModelNotLoadedError`, `ExtractionError`, `SSRFBlockedError`, etc.), never raw `HTTPException` — that translation happens in the router.

`ml/` — the model itself: `model_loader.py` (singleton, loaded once at app startup, not per-request), `preprocess.py` (imports `research/src/data/clean.py`, never reimplements it), `inference.py` (tokenize → forward pass → softmax → label/confidence).

## 2. Configuration (`app/config.py`, `pydantic-settings`)

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_REPO_ID` | Hugging Face Hub repo ID to load at startup | `<to be set once a checkpoint exists>` |
| `MODEL_REVISION` | Pinned Hub revision/commit for reproducibility (`REPRODUCIBILITY.md`) | `main` (tighten to a commit SHA once stable) |
| `ALLOWED_ORIGINS` | CORS allow-list | the deployed Vercel domain + `http://localhost:3000` for dev |
| `RATE_LIMIT_PER_MINUTE` | Requests per IP per minute | `20` |
| `MAX_TEXT_LENGTH` | Upper bound for submitted text | `5000` |
| `MIN_TEXT_LENGTH` | Lower bound | `10` |
| `EXTRACTION_TIMEOUT_SECONDS` | URL fetch timeout | `10` |
| `EXTRACTION_MAX_BYTES` | URL response size cap | `2_000_000` |
| `LOG_LEVEL` | Standard logging level | `INFO` |

All read via `pydantic-settings` from environment variables; `.env.example` documents names only, no values, per `SECURITY.md`.

## 3. Endpoints

### `POST /api/v1/analyze`

**Purpose:** predict misinformation-risk pattern for submitted text or a URL.
**Auth:** none. **Rate limit:** `RATE_LIMIT_PER_MINUTE` per IP.

Request:
```json
{ "text": "string, optional", "url": "string, optional" }
```
Exactly one of `text`/`url` must be present (validated in the Pydantic schema via a model validator).

Response `200`:
```json
{
  "id": "uuid",
  "label": "real_pattern" | "fake_pattern",
  "confidence": 0.87,
  "script_warning": false,
  "disclaimer": "This system provides an AI-assisted analysis of linguistic patterns... (full text from docs/responsible_ai.md)"
}
```

Errors: `400` (both/neither field present, text outside 10–5000 chars) → `{"detail": "..."}`; `422` (URL blocked by SSRF policy) → `{"detail": "This URL cannot be processed for security reasons."}`; `502` (extraction failed — page unreachable/unparseable) → `{"detail": "Could not extract article text from this URL."}`; `500` (model error).

Example request: `curl -X POST /api/v1/analyze -d '{"text": "..."}'`. Latency target: <2s (CPU inference, `ARCHITECTURE.md`).

### `POST /api/v1/explain`

**Purpose:** compute an Integrated-Gradients explanation for a prior `/analyze` response.
**Auth:** none. **Rate limit:** shared with `/analyze`.

Request: `{ "id": "uuid" }` — must reference an id returned by `/analyze` within the last N minutes (short-lived in-process cache of the original input text keyed by id — not a database, consistent with E8; cache size-bounded and TTL-expired to avoid unbounded memory growth).

Response `200`:
```json
{
  "id": "uuid",
  "spans": [{ "text": "string", "attribution": 0.42 }],
  "note": "These text features contributed most strongly to the model's prediction."
}
```

Errors: `404` (id not found/expired) → prompts the frontend to re-run `/analyze`; `500` (explanation computation error); `504` (explanation timeout, e.g. >8s) — frontend degrades gracefully per `FRONTEND_SPECIFICATION.md`.

### `GET /api/v1/examples`

**Purpose:** curated example articles for the demo. **Auth:** none. **Rate limit:** none (static, cheap).

Response `200`:
```json
{ "examples": [{ "id": "string", "title": "string", "text": "string", "source_dataset": "ax_to_grind" | "notri_fact" }] }
```
Served from the static `backend/app/data/examples.json` fixture (E8) — no database read.

### `GET /api/v1/model-info`

**Purpose:** transparency — model/version/training-data summary. **Auth:** none.

Response `200`:
```json
{
  "model": "xlm-roberta-base (length-stratified, mitigated checkpoint)",
  "version": "urdu-misinfo-xlmr-axtogrind-mitigated-v1",
  "training_data": "Ax-to-Grind Urdu (2017-2023, 15 domains)",
  "known_limitations": ["May be unreliable on satire or overtly political/religious content", "Confidence is not a probability of factual falsehood", "Trained data covers 2017-2023; emerging topics may be out of distribution"]
}
```

### `GET /api/v1/health`

**Purpose:** liveness/readiness. **Auth:** none. **Rate limit:** none.

Response `200`: `{ "status": "ok", "model_loaded": true }`. Response `503` if the model failed to load at startup.

## 4. Error taxonomy (used consistently across all endpoints)

| Status | Meaning | Example |
|---|---|---|
| `400` | Malformed/invalid client request | missing both `text`/`url`, out-of-bounds length |
| `404` | Referenced resource not found/expired | `/explain` id not found |
| `422` | Request well-formed but rejected by a safety policy | SSRF-blocked URL |
| `500` | Unexpected server/model error | model inference crash |
| `502` | Upstream dependency failure | URL fetch/extraction failed |
| `503` | Service not ready | model not loaded at startup |
| `504` | Operation timed out | explanation computation exceeded budget |

## 5. Logging

Request metadata only (timestamp, endpoint, input length, prediction label, confidence, latency) — **raw submitted text is never logged by default** (privacy stance, given plausible political/religious/sensitive input — `MASTER_PROJECT_BLUEPRINT.md` Part 18/19). An opt-in `DEBUG`-level flag enables verbose local-dev logging only, never enabled in the deployed environment.

## 6. Endpoint traceability (consumer / implementation / tests)

Closes the loop between this contract, the frontend, and the test suite — every endpoint below must remain consistent across all three columns; if one changes, update the others in the same commit (`CLAUDE.md` rule 8).

| Endpoint | Frontend consumer | Backend implementation | Tests |
|---|---|---|---|
| `POST /api/v1/analyze` | `components/analyze/AnalyzeForm.tsx` via `lib/hooks/useAnalyze.ts` → `lib/api-client.ts` | `routers/analyze.py` → `services/inference_service.py` (+ `services/extraction_service.py` if `url`) | `backend/tests/test_analyze.py`, `frontend/tests/unit/AnalyzeForm.test.tsx`, the Playwright E2E flow |
| `POST /api/v1/explain` | `components/results/ExplainButton.tsx` / `ExplanationView.tsx` via `lib/hooks/useExplain.ts` | `routers/explain.py` → `services/explainability_service.py` | `backend/tests/test_explain.py`, `frontend/tests/unit/ExplanationView.test.tsx` |
| `GET /api/v1/examples` | `components/analyze/ExampleArticlePicker.tsx` | `routers/meta.py` (reads `app/data/examples.json`) | `backend/tests/test_meta.py` |
| `GET /api/v1/model-info` | `/model` page | `routers/meta.py` | `backend/tests/test_meta.py` |
| `GET /api/v1/health` | Optional cold-start polling on `/analyze` (`FRONTEND_SPECIFICATION.md` Section 4) | `routers/meta.py` | `backend/tests/test_meta.py` |

## 7. Versioning

Path-based (`/api/v1/`). A breaking response-shape change requires `/api/v2/` with the old version kept alive during any transition — there is currently exactly one consumer (the frontend in this same repo), so this policy exists for discipline, not because multiple external consumers are expected yet.
