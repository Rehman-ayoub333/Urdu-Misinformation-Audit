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

**Provisional default: 384 subword tokens**, truncation from the end (Ax-to-Grind's real-article median is short per the literature; 384 comfortably covers most real articles while bounding the small number of very long fake articles central to the length-confound research question — capping too aggressively by default would itself interfere with RQ1/RQ2's undisturbed measurement, so this value is for **serving**, not for the RQ1–RQ3 training/evaluation runs, which use the full, uncapped Tier-2-cleaned text per `MASTER_PROJECT_BLUEPRINT.md` Part 8).

`DECISION REQUIRED` (`DECISION_REGISTER.md` U2): finalize this number once `research/data/audit/` produces the real subword-length distribution (Milestone 2, `ROADMAP.md`). Update this section and `app/config.py`'s default together when resolved.

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
