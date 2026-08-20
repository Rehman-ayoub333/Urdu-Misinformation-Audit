# Reproducibility

Expands `MASTER_PROJECT_BLUEPRINT.md` Part 25 into exact, implementable conventions.

## 1. Versions

Python 3.12 pinned (backend `pyproject.toml`, research `pyproject.toml`, and CI all target this exact minor version). All Python package versions pinned exactly (`==`, not `>=`) in `backend/requirements.txt` and `research/requirements.txt` — critically `torch`, `transformers`, `captum`, `scikit-learn`, `fastapi`. Node/npm versions pinned via `frontend/package.json`'s `engines` field and a committed lockfile (`package-lock.json`).

> **Pinning the direct dependency is not always enough on a hosted image.** Both GPU platforms ship a large preinstalled scientific stack, and `pip install` over it can leave a package in a state that is present but internally inconsistent — not merely the wrong version. Two cases are recorded and handled in `research/notebooks/04_transformer_training.ipynb`: torchvision, uninstalled because it is genuinely unused here (`DECISION_REGISTER.md` M4-3), and numpy/scipy, force-reinstalled because scipy is very much used — `sklearn.metrics`, which every metrics file goes through, pulls in 493 scipy modules (`DECISION_REGISTER.md` M4-4). "Pinned exactly" therefore means pinned *and* actually installed as pinned; the notebook's environment gate asserts the latter rather than assuming it.

> **This 3.12 pin supersedes the original 3.11 pin, per `DECISION_REGISTER.md` M4-2 (resolved 2026-08-15, option (b)).** 3.11 was unsatisfiable by construction: `numpy==2.5.2`, pinned in `research/requirements.txt`, requires ≥3.12 and ships no cp311 wheel, so the repo's own dependency set could never be installed on the interpreter the repo declared. 3.12 is also the version the GPU notebook environment provides (Colab and Kaggle both), so training and evaluation now share one interpreter with the classical baselines. Milestone 3's four classical experiments (A, B, H, H2) were **re-run on 3.12 and their committed metrics overwritten** as part of that resolution — every number in `research/results/metrics/` therefore traces to a single stated interpreter. Do not reintroduce a 3.11 target anywhere without a new decision row.

## 2. Seeds

Fixed set `{42, 123, 2026}` for every 3-seed transformer experiment (`EXPERIMENT_PLAN.md` Section 5); seed is a required field in every `research/configs/model_*.yaml` run — no experiment script accepts an unseeded run for a transformer model. Classical baselines (deterministic) don't need seed repetition but still record `random_state=42` explicitly in config rather than relying on a library default.

## 3. Dataset versioning

The SHA256 manifest from `DATASET_PLAN.md` step 1 (`research/data/raw/MANIFEST.sha256`) **is** the dataset version identifier — referenced in every experiment config and in `docs/dataset_card.md`. If an upstream source dataset is ever updated, the old manifest entry is preserved (not overwritten) and a new one added, so any past experiment's exact input data remains identifiable.

## 4. Config-driven experiments

Every experiment run is fully specified by a `research/configs/*.yaml` file (`EXPERIMENT_PLAN.md` Section 3) — no experiment is launched from ad hoc CLI flags whose values aren't captured anywhere. The exact config file used for a run is copied alongside that run's output in `research/results/metrics/` (e.g., `results/metrics/D_xlmr_seed42.json` is accompanied by `results/metrics/D_xlmr_seed42.config.yaml`), so a reviewer can see precisely what produced any given number without reverse-engineering it from code history.

## 5. Hardware

Training hardware documented explicitly in `research/results/metrics/` run metadata (auto-captured by the training script, not hand-typed): Google Colab T4 GPU, VRAM, and the PyTorch/CUDA version active at run time (numerics can shift subtly across CUDA versions — worth capturing even if it's rarely the actual explanation for a discrepancy).

## 6. Saved artifacts

Every experiment saves: the metrics JSON (accuracy, macro-F1, etc.), the config used, and — for the test and cross-dataset test sets specifically — the **full per-example predictions** (not just aggregate metrics), so any derived statistic (e.g., a different metric someone wants to compute later, or a finer-grained length-bucket breakdown) can be recomputed without rerunning inference.

**Format of the per-example predictions** (added `DECISION_REGISTER.md` M5-2 — this requirement was stated here from the start but never specified or implemented, and the gap surfaced when Milestone 5's error-analysis sample needed it):

- **One JSONL file per metrics file**, named by replacing `.json` with `.predictions.jsonl` — e.g. `A_tfidf_logreg_ax_to_grind_test.json` is accompanied by `A_tfidf_logreg_ax_to_grind_test.predictions.jsonl`. The pair is therefore obvious on disk without a lookup table.
- **JSONL, not CSV or parquet**: one self-describing object per line, so it diffs readably in git and streams without loading the whole file, and it needs no dependency beyond the standard library (parquet would mean adding `pyarrow`, which `CLAUDE.md` rule 16 does not permit without a stated need).
- **Fields per row:** `row_id` (the identifier from the committed split index, so the source text is recoverable via `research/src/data/split.py` rather than by position), `y_true`, `y_pred`, and `score_positive` where the model produces one. `score_positive` is P(positive class); the positive label is *not* repeated per row — it is recorded once in the sibling metrics JSON at `config.data.labels.positive`.
- **Scores are written at full precision, never rounded.** Rounding would silently cap how faithfully a different metric can be recomputed later, which is the entire purpose of keeping these.
- **Which splits:** `test` and `holdout` only, defined once as `PREDICTION_SPLITS` in `research/src/evaluation/metrics.py` so an experiment runner cannot quietly skip them. Validation splits are deliberately excluded — this section scopes the requirement to test sets, and val predictions drive model selection rather than reported findings.
- **`LinearSVC` has no probabilities**, so `score_positive` is absent for experiment B rather than fabricated. Model checkpoints themselves are not stored in git (too large) — they live on the HF Model Hub (`DEPLOYMENT_PLAN.md` Section 3), referenced by repo ID + revision from the metrics metadata.

## 7. Notebooks

`research/notebooks/` is exploration-only. Any result that matters for the thesis or the deployed model must exist as a script-produced artifact in `research/results/`, not only inside a notebook's transient cell output (`CLAUDE.md` rule 2, `MASTER_PROJECT_BLUEPRINT.md` Part 20's closing rule). Notebooks may call the same `research/src/` functions used by the scripts, but must not contain independent, undocumented logic that produces a number quoted elsewhere.

## 8. What "reproducible" means for this project, concretely

Another researcher with access to this repo should be able to: run `research/scripts/download_data.py` and get byte-identical raw files (checksum-verified against the committed manifest); run the REQUIRED experiment set (`EXPERIMENT_PLAN.md`) from `research/configs/experiment_matrix.yaml` and get metrics within the documented seed-variance band of the ones committed in `research/results/metrics/`; and load the exact deployed checkpoint from the HF Hub using the pinned `MODEL_REVISION` in `DEPLOYMENT_PLAN.md`'s env var list.
