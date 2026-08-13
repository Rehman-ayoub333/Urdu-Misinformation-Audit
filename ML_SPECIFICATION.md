# ML Specification

Covers the serving-side inference pipeline in `backend/app/ml/`. For the research/training pipeline (data audit, experiment matrix, model training), see `DATASET_PLAN.md` and `EXPERIMENT_PLAN.md` — this document is specifically about how a trained checkpoint is loaded and served in production.

## 1. Pipeline stages

```
Frontend submits text/URL
  → (if URL) ExtractionService.extract(url) → raw article text
  → PreprocessService.clean(text)            # backend/app/ml/preprocess.py, imports research/src/data/clean.py — Tier 2 cleaning ONLY, identical function used at training time
  → tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
  → model.forward(**inputs)  → logits
  → softmax(logits) → { real_pattern: p0, fake_pattern: p1 }
  → label = argmax; confidence = max(p0, p1)
  → (calibration note below)
  → response assembled with the standing disclaimer (docs/responsible_ai.md)

Separately, on-demand:
  → ExplainabilityService.explain(text, predicted_label)
  → IntegratedGradients(model, baseline=all-pad-token input) → token-level attributions
  → aggregate subword attributions back to word-level spans for display
  → response with spans + fixed caption
```

## 2. Model loading

Loaded **once**, at FastAPI startup (`app/main.py`'s lifespan handler), from `MODEL_REPO_ID`/`MODEL_REVISION` (`BACKEND_SPECIFICATION.md` config) via `transformers.AutoModelForSequenceClassification.from_pretrained` + `AutoTokenizer.from_pretrained`. Never reloaded per-request. If loading fails, `/api/v1/health` reports `model_loaded: false` and `analyze`/`explain` return `503` rather than crashing the process.

**Device selection:** CPU in production (HF Spaces free CPU tier — `DEPLOYMENT_PLAN.md`). The code must not hardcode `.cuda()` — use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` so the exact same code path works for local GPU development and CPU production without a branch.

**Explainability model:** the same loaded model instance is reused for Integrated Gradients (no second copy loaded) — `captum`'s `LayerIntegratedGradients` wraps the existing model's embedding layer.

## 3. Sequence length

**RESOLVED at Milestone 2: `MAX_SEQ_LEN = 512` subword tokens**, truncation from the end. This supersedes the provisional 384 (`DECISION_REGISTER.md` U2 → M2-1). This value governs **serving only** — the RQ1–RQ3 training/evaluation runs use the full, uncapped Tier-2-cleaned text per `MASTER_PROJECT_BLUEPRINT.md` Part 8, because capping by default would pre-empt the very confound RQ3 measures.

**Evidence** (`research/data/audit/dataset_quality_report.json`, real `xlm-roberta-base` tokenizer over Tier-2-cleaned text):

| Dataset | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| Ax-to-Grind (n=10,083) | 40 | 200 | 496 | 1,045 | 1,447 |
| Notri-Fact (n=13,388) | 201 | 306 | 349 | 482 | 2,217 |

Share of articles that fit **without truncation**:

| max_length | Ax-to-Grind | Notri-Fact | combined |
|---|---|---|---|
| 256 | 91.5% | 76.6% | 83.0% |
| 384 (old provisional) | 93.8% | 97.0% | 95.6% |
| **512 (chosen)** | **95.3%** | **99.2%** | **97.5%** |

**Why 512.** It is XLM-R's architectural maximum (512 learned position embeddings), so no larger value is available. Serving input is a user-pasted news article, which resembles Notri-Fact's full articles far more than Ax-to-Grind's headline-length snippets (median 40 tokens — Ax-to-Grind is largely headlines, not article bodies), making Notri-Fact's 99.2% the more representative coverage figure. Moving 384 → 512 costs nothing architecturally and recovers the 384–512 band, which is densely populated in exactly the full-article shape real users submit.

**Cost and fallback.** 512 raises per-request attention cost roughly 1.8× over 384 on the free-tier CPU. If measured latency at Milestone 9 breaches `BACKEND_SPECIFICATION.md`'s <2 s target, the documented order of remedies is (1) resolve `DECISION_REGISTER.md` U4 (quantization) — the lever intended for exactly this, then (2) fall back to 384, accepting 95.6% combined coverage. Reducing below 384 is not acceptable: 256 truncates 23% of full-article-shaped input.

Keep `app/config.py`'s `MAX_TEXT_LENGTH` default in sync with this number when the backend is implemented (Milestone 6).

## 4. Non-Urdu / script detection

A lightweight heuristic (not a full language-ID model — unjustified complexity for a soft warning) checks the proportion of Arabic-script Unicode codepoints in the submitted text. Below a threshold (e.g., <30% Arabic-script characters), the response sets `"script_warning": true` but **still returns a prediction** — the system degrades to "here's a result, but treat it skeptically," not a hard refusal, consistent with the responsible-AI framing (disclose uncertainty, don't pretend a false boundary).

## 5. Calibration

No post-hoc calibration (e.g., temperature scaling) is applied by default in the MVP — the raw softmax confidence is shown, always labeled "model confidence," never "probability of being false" (`MASTER_PROJECT_BLUEPRINT.md` Part 16, enforced in `docs/responsible_ai.md` copy). The reliability-diagram/ECE check from `MASTER_PROJECT_BLUEPRINT.md` Part 13 is a **research** deliverable (goes in `research/results/figures/`, feeds the Model chapter/page) — if it reveals the model is badly miscalibrated, that finding is reported honestly on `/model`, and temperature scaling becomes a candidate follow-up, not something silently applied to hide poor calibration.

## 6. Truncation and long-input handling

Server-side: the tokenizer truncates to `MAX_SEQ_LEN`. Client-side: `FRONTEND_SPECIFICATION.md`'s hard 5000-character cap prevents pathologically large requests from ever reaching the backend. These are two independent layers (defense in depth) — the backend must not assume the frontend's cap was respected (a direct API call could bypass it).

## 7. Timeout and error handling

Inference (`analyze`): no explicit timeout needed beyond the request's overall HTTP timeout — a single forward pass on XLM-R-base/CPU is fast (target <2s, `BACKEND_SPECIFICATION.md`). Explanation (`explain`): explicit internal timeout (~8s) since Integrated Gradients runs multiple forward/backward passes; on timeout, return `504` rather than hanging the request. Any unhandled exception in `ml/inference.py` or `ml/explainability`  is caught at the service layer and re-raised as a typed exception the router maps to `500` — raw stack traces are never returned in the API response (logged server-side only).

## 8. Batching

Not implemented in the MVP. The public demo serves one request at a time per model call; given the free-tier CPU deployment and expected low concurrent traffic, request-level batching would add complexity (queuing logic) without a measured need. Revisit only if `DEPLOYMENT_PLAN.md`'s load testing shows it's actually a bottleneck.

## 9. Preprocessing parity (the single most important engineering rule in this spec)

`backend/app/ml/preprocess.py` **must** import and call `research/src/data/clean.py`'s functions directly — it must not contain a second implementation of Unicode normalization, whitespace handling, etc. This is enforced by a specific ML sanity test in `TESTING_STRATEGY.md` (`test_ml_sanity.py::test_preprocessing_parity`) that asserts the serving-side and training-side cleaning functions are the literal same function object, not just "produce similar output" — any refactor that breaks this import must fail CI.
