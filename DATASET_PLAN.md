# Dataset Plan (Operational)

Full rationale and comparison matrix: `MASTER_PROJECT_BLUEPRINT.md` Parts 6–9. This document translates that design into the exact scripts/commands/gates Claude Code implements in `research/`.

## 1. Datasets and roles (final, per `DECISION_REGISTER.md` R2/R3)

| Dataset | Role | Source |
|---|---|---|
| Ax-to-Grind Urdu (10,083 articles) | Primary training set | `github.com/Sheetal83/Ax-to-Grind-Urdu-Dataset` — **the pipeline reads `Combined .csv`** (see below) |
| Notri-Fact Urdu (13,388 articles) | Cross-dataset test set (never trained on) | Kaggle: `tridata/notri-fact-real-and-unreal-urdu-news` (single `.xlsx` workbook) |

**Ax-to-Grind source file (`DECISION_REGISTER.md` M1-3, approved).** This table originally named `Fake News.csv` and `True News.csv`. That was wrong about the upstream repository's contents, which Milestone 1 measured and independently re-verified:

| File | Read by the pipeline? | State |
|---|---|---|
| `Combined .csv` | **Yes — sole text source** | Comma-separated, UTF-8+BOM, intact. 5,053 FAKE + 5,030 TRUE = 10,083 rows. |
| `True News.csv` | Cross-check only | Intact, but the TRUE half only. `validate.py::cross_check_true_news` asserts its 5,030 rows equal `Combined .csv`'s TRUE rows as a multiset — it is never a text source. |
| `Fake News.csv` | No — evidence only | **Corrupt upstream.** Despite its name it holds the full dataset, and a lossy encoding conversion replaced every Urdu character with a literal `?` (2,694,909 of 3,664,112 bytes, 73.5%). `validate.py::check_fake_news_still_corrupt` asserts it is *still* broken, so a silent upstream fix is noticed rather than changing the dataset unannounced. |

Fake-labelled Urdu text exists **only** in `Combined .csv`; reading the two originally-named files would yield zero usable fake training text.

**Known raw-file issues, handled explicitly in `validate.py` (not left to surface downstream):** 22 fully-blank trailing rows (raw indices 10084–10105) and 1 stray repeated header row at raw index 5053 — the seam where the FAKE and TRUE blocks were concatenated — are dropped with logged counts, taking 10,106 raw rows to exactly the documented 10,083. 47 labels carrying trailing whitespace (`"TRUE "` ×33, `"FAKE "` ×14) are normalised rather than dropped, since their intent is unambiguous. Note also that `Sr. No.` restarts at 1 for the TRUE block, so it has 5,051 duplicate values and is **not** usable as a row identifier; `validate.py` assigns stable `axg-#####` / `ntf-#####` ids from raw-file position instead, which is what makes the committed split index files reproducible.
| UFND (14,178 articles) | Optional secondary — `DECISION REQUIRED` U1 | Link not yet located; attempt via the paper's Data Availability Statement in Milestone 1 |

## 2. Pipeline (`research/src/data/`, executed in this order — matches `MASTER_PROJECT_BLUEPRINT.md` Part 20)

1. **`download.py`** — fetch raw CSVs from source URLs above into `research/data/raw/{ax_to_grind,notri_fact}/`; compute and store a SHA256 manifest (`research/data/raw/MANIFEST.sha256`) for every file. This manifest is the dataset's reproducibility anchor (`REPRODUCIBILITY.md`).
2. **`validate.py`** — schema check: expected columns present, exactly two label values, no null text fields, row count matches the documented size (10,083 / 13,388) within a small tolerance (source files occasionally get minor corrections upstream — an exact mismatch should warn, not silently pass, but a small delta shouldn't hard-fail the pipeline).
3. **`clean.py`** (Tier 2) — Unicode NFC normalization, Urdu/Arabic character-variant normalization (yeh/kaf variants), whitespace collapsing, HTML entity/tag stripping, URL/emoji → placeholder tokens (never deleted outright — see `MASTER_PROJECT_BLUEPRINT.md` Part 8 for the "don't over-clean" rationale). **This module is imported by `backend/app/ml/preprocess.py` — it is the single source of truth for cleaning logic, per `ML_SPECIFICATION.md` Section 9.**
4. **`dedup.py`** — exact-duplicate (hash) and near-duplicate (similarity-threshold) detection, run (a) within Ax-to-Grind, (b) within Notri-Fact, and (c) **across** Ax-to-Grind and Notri-Fact. Step (c) is a **go/no-go gate**: if any cross-dataset near-duplicates are found, they must be removed from Notri-Fact (the test side) before any cross-dataset experiment (`EXPERIMENT_PLAN.md` F/G) is run — this gate blocks Milestone 3 until it passes, per `MASTER_PROJECT_BLUEPRINT.md` Part 35's risk register.
5. **`audit.py`** — produces `research/data/audit/`: length distributions (char/word/subword-tokenizer counts) by label, per-dataset; top vocabulary and log-odds word association by label; source/domain cross-tabs where a source field exists; structural-artifact scan (boilerplate/byline detection); language-mixing proportion. Outputs both raw CSV/JSON and the figures listed in `MASTER_PROJECT_BLUEPRINT.md` Part 31 (items 1–3). This is where `ML_SPECIFICATION.md`'s provisional 384-token sequence-length default gets confirmed or revised.
6. **`split.py`** — stratified 70/15/15 train/val/test on Ax-to-Grind (dedup'd), with grouped splitting by source/publisher if `audit.py` found a usable source field (decide from evidence, not by default). Notri-Fact is held out **in full** as the cross-dataset test set — never split, never trained on. Split membership is stored as row-ID index files in `research/data/splits/` (committed to git — small, and this is exactly what makes the split reproducible for another researcher).

## 3. Immutability rule

`research/data/raw/` is never modified in place after `download.py` runs — if a fix to raw data is ever needed, it happens by re-running `download.py` against a newly pinned source and recording a new manifest entry, not by hand-editing files. `research/data/clean/` and `research/data/processed/` are fully regenerable from `raw/` via the pipeline scripts and are gitignored; only `research/data/splits/` and `research/data/audit/` outputs are committed (they're small and are core evidence, per `ARCHITECTURE.md`'s repo tree).

## 4. UFND resolution (`DECISION REQUIRED` U1)

During Milestone 1, attempt to locate UFND's actual download link via the paper's Data Availability Statement (`nature.com/articles/s41598-026-36771-0`). If found: add `research/data/raw/ufnd/` and extend `download.py`/`validate.py`/`audit.py` to cover it, and update `DECISION_REGISTER.md` with the resolved link. If not found within Milestone 1's timebox: log the attempt and outcome in `DECISION_REGISTER.md`, drop UFND from the active plan, and proceed with Ax-to-Grind + Notri-Fact only — this must not block subsequent milestones.
