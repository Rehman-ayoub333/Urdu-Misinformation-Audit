# Testing Strategy

Tooling: pytest + httpx TestClient (backend), Vitest + React Testing Library + Playwright (frontend) — `DECISION_REGISTER.md` E13/E14. This document lists actual test files and what each must assert, so Claude Code writes real, specific tests rather than generic smoke tests.

## 1. Backend (`backend/tests/`)

| File | Covers |
|---|---|
| `test_analyze.py` | `POST /api/v1/analyze`: valid text → 200 with correct schema; valid URL → 200; both fields present → 400; neither field → 400; text <10 or >5000 chars → 400; malformed JSON → 422 (FastAPI default); confirms `disclaimer` field is always non-empty and matches `docs/responsible_ai.md`'s text exactly (a literal string-equality assertion, not just "field exists") |
| `test_explain.py` | Valid prior id → 200 with spans; unknown/expired id → 404; explanation timeout path → 504 (mocked) |
| `test_meta.py` | `/health` returns `model_loaded: true` when a model is loaded (mocked in test), `503` when not; `/examples` returns the static fixture; `/model-info` returns all required fields |
| `test_ssrf.py` | **Security regression tests, not optional**: requests to `127.0.0.1`, `169.254.169.254` (cloud metadata endpoint — a classic real-world SSRF target), `10.0.0.0/8` range, non-standard ports, and a redirect chain that resolves to a private IP are all rejected with `422`; a normal public news URL passes validation (mocked DNS/fetch) |
| `test_ml_sanity.py` | Model output probabilities sum to 1 (within float tolerance); inference is deterministic given a fixed seed + identical input; empty string is rejected before reaching the model; a 10,000-character input is truncated, not crashed on; a saved checkpoint loads without error and reproduces a fixed regression-test prediction; **`test_preprocessing_parity`**: asserts `backend.app.ml.preprocess.clean` is the same function object as (or a direct thin wrapper with identical output on a fixed test-string battery for) `research.src.data.clean.clean_text` — this is the specific test protecting `ML_SPECIFICATION.md` Section 9's parity rule; IG attribution spans sum to a value consistent with the completeness axiom within tolerance; **`test_non_urdu_script_warning`**: a fixture English-only input still returns a `200` with a real prediction and `script_warning: true` (per `ML_SPECIFICATION.md` Section 4 — a soft warning, never a hard block); **`test_urdu_text_roundtrip`**: a fixture Urdu-script input with mixed Urdu/English code-switching and Eastern Arabic-Indic numerals survives `preprocess.clean` without corruption (a direct regression test for the Unicode-normalization rules in `DATASET_PLAN.md` Section 2 step 3, run against real Urdu text, not only Latin-script fixtures — the risk being that a preprocessing bug could pass every test if the test suite itself never exercises real Urdu input) |
| `conftest.py` | Shared fixtures: a `TestClient` instance, a mocked/small model for fast test runs (full checkpoint loading is too slow for the unit-test suite — a tiny dummy model with the same interface is used, with at least one test in `test_ml_sanity.py` explicitly running against the real checkpoint as a separate, slower "integration" marker) |

**Data validation tests** (in `research/`, run alongside the backend suite or as their own pytest target): `research/data/raw` schema checks (`DATASET_PLAN.md` step 2) expressed as real pytest assertions, not just print statements in a notebook — e.g., `test_ax_to_grind_has_two_labels()`, `test_no_null_text_after_cleaning()`.

## 2. Frontend (`frontend/tests/unit/`, Vitest + RTL)

Component-level tests for state transitions, matching `FRONTEND_SPECIFICATION.md`'s documented states exactly: `AnalyzeForm` (idle → loading → success/error, disabled-button logic for invalid input), `PredictionCard` (renders label/confidence/disclaimer together — a test that fails if the disclaimer is ever omitted, since `CLAUDE.md` rule 14 makes this a hard requirement, not a nice-to-have), `ExplanationView` (renders the fixed caption text verbatim), `ArticleInput` (RTL `dir` attribute applied when Urdu-script text is present), `ExampleArticlePicker` (loading skeleton, silent-hide-on-error behavior).

## 3. End-to-end (`frontend/tests/e2e/`, Playwright, Chromium only)

**One required flow**, per `MASTER_PROJECT_BLUEPRINT.md` Part 24: load `/analyze` → select a curated example article → click Analyze → assert the prediction renders → click "See why" → assert the explanation renders. Run against a locally-running backend with the real (or a small test) checkpoint in CI. This is the only E2E test required for the MVP — additional E2E coverage (URL flow, error flows) is a stretch addition, not a blocker, given the disproportionate CI-time cost of a large E2E suite for a solo-developer project (`MASTER_PROJECT_BLUEPRINT.md` Part 24's own reasoning, carried forward).

## 4. Security testing

Covered primarily by `test_ssrf.py` above, plus: oversized-request rejection (a >2MB URL response is rejected, tested with a mocked oversized response), HTML/script content in extracted article text is stripped before reaching the model (tested against a fixture HTML page containing a `<script>` tag), rate-limit enforcement (N+1th request within a minute from the same IP returns `429`, tested against the `slowapi` middleware directly rather than via real IP spoofing).

## 5. Acceptance criteria (ties back to `ROADMAP.md`)

A milestone touching `backend/` is not complete until `pytest` is green and `ruff`/`mypy` pass. A milestone touching `frontend/` is not complete until `npm run test` (Vitest) and `npx playwright test` are green and `eslint`/`tsc --noEmit` pass. A milestone touching `research/` is not complete until its data-validation tests pass **and** its REQUIRED experiments (`EXPERIMENT_PLAN.md`) have produced real metrics files in `research/results/metrics/` — a green test suite alone does not certify a research milestone, since tests check code correctness, not scientific validity of results (that's what the audit gates and multi-seed reporting in `EXPERIMENT_PLAN.md` are for).
