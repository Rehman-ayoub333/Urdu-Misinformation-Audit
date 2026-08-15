"""Load and schema-validate the raw datasets.

Implements `DATASET_PLAN.md` Section 2 step 2. This module owns two things:

1. **Which raw file each dataset is actually read from** (`DECISION_REGISTER.md` M1-3),
2. **The canonical in-memory schema** every downstream stage consumes.

Nothing downstream reads `research/data/raw/` directly — `clean.py`, `dedup.py`,
`audit.py` and `split.py` all go through `load_ax_to_grind()` / `load_notri_fact()`,
so the M1-3 source decision is made in exactly one place.

Canonical schema returned by both loaders:

    row_id  str   stable, dataset-prefixed, derived from the row's position in the
                  RAW file (e.g. "axg-00042"). Survivor rows keep their original
                  id when junk rows are dropped, so ids stay stable regardless of
                  what filtering runs — this is what makes the committed split
                  index files in research/data/splits/ reproducible.
    text    str   the article text, exactly as in the raw file. NOT cleaned here;
                  cleaning is clean.py's job (Tier 2, MASTER_PROJECT_BLUEPRINT.md
                  Part 8) and stays a separate, separately-testable stage.
    label   str   normalised to LABEL_REAL / LABEL_FAKE.

`Sr. No.` is deliberately NOT used as the row id: it restarts at 1 for the TRUE
block inside `Combined .csv`, so it has 5,051 duplicate values and cannot identify
a row.

Usage:
    python -m research.src.data.validate                     # both datasets
    python -m research.src.data.validate --only ax_to_grind  # one dataset
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from research.src.data.download import RAW_DIR

# --- Source-of-truth file selection (DECISION_REGISTER.md M1-3) ----------------

AX_TO_GRIND_DIR = RAW_DIR / "ax_to_grind"

# The ONLY Ax-to-Grind file the pipeline reads for text. `Fake News.csv` had its
# Urdu destroyed upstream by a lossy encoding conversion (73.5% of its bytes are
# the literal character '?'), and `True News.csv` holds only the TRUE half, so
# `Combined .csv` is the sole source containing intact text for both labels.
AX_TO_GRIND_SOURCE = AX_TO_GRIND_DIR / "Combined .csv"

# Read ONLY by cross_check_true_news(): an independent confirmation that the TRUE
# rows in Combined .csv match the standalone TRUE file. Never a text source.
AX_TO_GRIND_TRUE_CROSSCHECK = AX_TO_GRIND_DIR / "True News.csv"

# Read ONLY by check_fake_news_still_corrupt(). Retained as the provenance record
# of the corruption so that a future re-download which silently "fixes" it upstream
# is noticed rather than quietly changing the dataset under the project's feet.
AX_TO_GRIND_CORRUPT_EVIDENCE = AX_TO_GRIND_DIR / "Fake News.csv"

NOTRI_FACT_SOURCE = RAW_DIR / "notri_fact" / "Notri-Fact_Real_Unreal_Urdu_NEWS.xlsx"

# --- Canonical labels ---------------------------------------------------------

LABEL_REAL = "real"
LABEL_FAKE = "fake"
CANONICAL_LABELS = (LABEL_FAKE, LABEL_REAL)

# Raw label strings are matched after strip()+casefold(). Ax-to-Grind uses
# TRUE/FAKE (with stray trailing spaces on 47 rows); Notri-Fact uses Real/Unreal.
_LABEL_MAP = {
    "true": LABEL_REAL,
    "real": LABEL_REAL,
    "fake": LABEL_FAKE,
    "unreal": LABEL_FAKE,
}

# Documented sizes (DATASET_PLAN.md Section 1). Step 2 requires a warn-not-fail
# tolerance: upstream files occasionally get minor corrections, so an exact
# mismatch should be visible without hard-failing the pipeline.
DOCUMENTED_SIZES = {"ax_to_grind": 10_083, "notri_fact": 13_388}
SIZE_TOLERANCE = 0.01  # 1%

# The corruption signature of Fake News.csv, measured at Milestone 1 and
# independently re-verified (DECISION_REGISTER.md M1-3).
CORRUPT_FAKE_NEWS_SHA256 = "7824493c623a720058d6aca19d9701b1d7077ce950026a8126dee725904f3de1"
CORRUPT_FAKE_NEWS_QUESTION_MARK_BYTES = 2_694_909


class ValidationError(RuntimeError):
    """Raised when a dataset violates a schema guarantee downstream code relies on."""


@dataclass
class LoadReport:
    """What the loader dropped and why. Printed and asserted on — never silent."""

    dataset: str
    raw_rows: int
    dropped_blank: int = 0
    dropped_stray_header: int = 0
    dropped_unmappable_label: int = 0
    final_rows: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total_dropped(self) -> int:
        return self.dropped_blank + self.dropped_stray_header + self.dropped_unmappable_label

    def render(self) -> str:
        lines = [
            f"[{self.dataset}] raw rows            : {self.raw_rows:,}",
            f"[{self.dataset}] dropped blank/null  : {self.dropped_blank:,}",
            f"[{self.dataset}] dropped stray header: {self.dropped_stray_header:,}",
            f"[{self.dataset}] dropped bad label   : {self.dropped_unmappable_label:,}",
            f"[{self.dataset}] final rows          : {self.final_rows:,}",
            f"[{self.dataset}] label counts        : {self.label_counts}",
        ]
        lines.extend(f"[{self.dataset}] note: {n}" for n in self.notes)
        return "\n".join(lines)


def _normalise_label(value: object) -> str | None:
    """Map a raw label cell to a canonical label, or None if unmappable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return _LABEL_MAP.get(str(value).strip().casefold())


def _assign_row_ids(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Stable ids from position in the RAW file, before any row is dropped."""
    return pd.Series([f"{prefix}-{i:05d}" for i in range(len(frame))], index=frame.index)


def load_ax_to_grind(*, source: Path | None = None) -> tuple[pd.DataFrame, LoadReport]:
    """Load Ax-to-Grind from `Combined .csv` into the canonical schema.

    Known raw-file issues, all handled explicitly here rather than being left to
    surface as mysterious NaNs somewhere downstream:

    * **22 fully-blank trailing rows** (raw indices 10084-10105) — every column NaN.
      Export padding. Dropped, counted.
    * **1 stray repeated header row** at raw index 5053, the seam where the FAKE
      block (5,053 rows) and the TRUE block (5,030 rows) were concatenated. Its
      cells literally read "Sr. No." / "News Items" / "Label". Dropped, counted.
    * **47 labels with trailing whitespace** ("TRUE " ×33, "FAKE " ×14) — normalised
      rather than dropped, since the intent is unambiguous.

    Dropping is the right call for the first two: they carry no article text at all,
    so there is nothing to recover, and keeping them would put empty strings into the
    model's training data.
    """
    path = source or AX_TO_GRIND_SOURCE
    if not path.exists():
        raise ValidationError(
            f"{path} is missing. Run `python -m research.src.data.download` first."
        )

    # utf-8-sig: the file carries a BOM. Everything is read as string; no dtype
    # inference, so a numeric-looking article can never become a float.
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=True)

    expected_columns = ["Sr. No.", "News Items", "Label"]
    if list(raw.columns) != expected_columns:
        raise ValidationError(
            f"{path.name}: unexpected columns {list(raw.columns)!r}, expected {expected_columns!r}"
        )

    report = LoadReport(dataset="ax_to_grind", raw_rows=len(raw))
    frame = raw.copy()
    frame["row_id"] = _assign_row_ids(frame, "axg")

    text = frame["News Items"].fillna("").map(str)
    label_raw = frame["Label"].fillna("").map(str)

    is_stray_header = (label_raw.str.strip() == "Label") & (
        frame["News Items"].fillna("").map(str).str.strip() == "News Items"
    )
    report.dropped_stray_header = int(is_stray_header.sum())
    frame = frame.loc[~is_stray_header]
    text = text.loc[frame.index]
    label_raw = label_raw.loc[frame.index]

    is_blank = text.str.strip() == ""
    report.dropped_blank = int(is_blank.sum())
    frame = frame.loc[~is_blank]

    frame["text"] = frame["News Items"].map(str)
    frame["label"] = frame["Label"].map(_normalise_label)

    unmappable = frame["label"].isna()
    report.dropped_unmappable_label = int(unmappable.sum())
    if report.dropped_unmappable_label:
        bad = sorted({str(v) for v in frame.loc[unmappable, "Label"].tolist()})[:5]
        report.notes.append(f"unmappable label values dropped, samples: {bad}")
    frame = frame.loc[~unmappable]

    result = frame[["row_id", "text", "label"]].reset_index(drop=True)
    report.final_rows = len(result)
    report.label_counts = result["label"].value_counts().to_dict()
    return result, report


def load_notri_fact(*, source: Path | None = None) -> tuple[pd.DataFrame, LoadReport]:
    """Load Notri-Fact from its Kaggle .xlsx workbook into the canonical schema.

    `News_Text` is the article body; `Headline`, `Category`, `Date` and `News_length`
    are carried through as extra columns because `audit.py` needs Category for the
    domain cross-tab and the domain-shift check. Milestone 1 found this file clean
    (zero nulls, no corruption), but the same checks run anyway — "it was clean last
    time" is not a validation.
    """
    path = source or NOTRI_FACT_SOURCE
    if not path.exists():
        raise ValidationError(
            f"{path} is missing. Run `python -m research.src.data.download --only notri_fact`."
        )

    raw = pd.read_excel(path, engine="openpyxl", dtype=str)

    expected_columns = [
        "Index",
        "Headline",
        "News_Text",
        "Category",
        "Date",
        "News_length",
        "Label",
    ]
    missing = [c for c in expected_columns if c not in raw.columns]
    if missing:
        raise ValidationError(f"{path.name}: missing expected columns {missing!r}")

    report = LoadReport(dataset="notri_fact", raw_rows=len(raw))
    frame = raw.copy()
    frame["row_id"] = _assign_row_ids(frame, "ntf")

    text = frame["News_Text"].fillna("").map(str)
    is_blank = text.str.strip() == ""
    report.dropped_blank = int(is_blank.sum())
    frame = frame.loc[~is_blank]

    frame["text"] = frame["News_Text"].map(str)
    frame["label"] = frame["Label"].map(_normalise_label)

    unmappable = frame["label"].isna()
    report.dropped_unmappable_label = int(unmappable.sum())
    if report.dropped_unmappable_label:
        bad = sorted({str(v) for v in frame.loc[unmappable, "Label"].tolist()})[:5]
        report.notes.append(f"unmappable label values dropped, samples: {bad}")
    frame = frame.loc[~unmappable]

    keep = ["row_id", "text", "label", "Headline", "Category", "Date", "News_length"]
    result = frame[keep].reset_index(drop=True)
    report.final_rows = len(result)
    report.label_counts = result["label"].value_counts().to_dict()
    return result, report


# --- Schema assertions --------------------------------------------------------


def validate_schema(frame: pd.DataFrame, dataset: str) -> None:
    """Assert the canonical guarantees every downstream stage relies on."""
    for column in ("row_id", "text", "label"):
        if column not in frame.columns:
            raise ValidationError(f"{dataset}: canonical column {column!r} is missing")

    if frame["row_id"].duplicated().any():
        dupes = frame.loc[frame["row_id"].duplicated(), "row_id"].head(5).tolist()
        raise ValidationError(f"{dataset}: row_id is not unique, e.g. {dupes}")

    n_null_text = int(frame["text"].isna().sum())
    n_blank_text = int(frame["text"].fillna("").map(str).str.strip().eq("").sum())
    if n_null_text or n_blank_text:
        raise ValidationError(
            f"{dataset}: {n_null_text} null and {n_blank_text} blank text values "
            "survived loading — the known-issue handling in the loader did not catch them."
        )

    labels = set(frame["label"].unique())
    if labels != set(CANONICAL_LABELS):
        raise ValidationError(
            f"{dataset}: expected exactly the labels {set(CANONICAL_LABELS)!r}, got {labels!r}"
        )


def check_documented_size(frame: pd.DataFrame, dataset: str) -> str:
    """Compare against DATASET_PLAN.md's documented size. Warns, never raises."""
    expected = DOCUMENTED_SIZES[dataset]
    actual = len(frame)
    delta = actual - expected
    within = abs(delta) <= expected * SIZE_TOLERANCE
    verdict = "OK" if delta == 0 else ("within tolerance" if within else "OUT OF TOLERANCE")
    return (
        f"[{dataset}] documented {expected:,}, got {actual:,} "
        f"(delta {delta:+,}, {verdict})"
    )


# --- M1-3 cross-check and provenance check ------------------------------------


def cross_check_true_news() -> tuple[bool, str]:
    """Confirm `True News.csv`'s rows really are Combined .csv's TRUE rows.

    This is the independent check M1-3 requires: `Combined .csv` became the sole
    text source on the strength of a byte-level corruption analysis, so its TRUE
    half is verified against the standalone TRUE file rather than assumed correct.

    Compares as multisets of whitespace-normalised text, because the two files
    differ in line endings and one carries a BOM — a raw string comparison would
    fail for reasons that have nothing to do with content.
    """
    if not AX_TO_GRIND_TRUE_CROSSCHECK.exists():
        return False, f"cross-check skipped: {AX_TO_GRIND_TRUE_CROSSCHECK.name} is missing"

    standalone = pd.read_csv(AX_TO_GRIND_TRUE_CROSSCHECK, encoding="utf-8-sig", dtype=str)
    standalone_text = (
        standalone["News Items"].fillna("").map(str).str.split().str.join(" ").pipe(
            lambda s: s[s != ""]
        )
    )

    combined, _ = load_ax_to_grind()
    combined_true = combined.loc[combined["label"] == LABEL_REAL, "text"]
    combined_true_text = combined_true.map(str).str.split().str.join(" ")

    left = sorted(standalone_text.tolist())
    right = sorted(combined_true_text.tolist())
    identical = left == right

    if identical:
        return True, (
            f"cross-check PASSED: True News.csv's {len(left):,} rows are exactly the "
            f"{len(right):,} TRUE rows in Combined .csv (multiset equality on "
            "whitespace-normalised text)"
        )

    only_standalone = len(set(left) - set(right))
    only_combined = len(set(right) - set(left))
    return False, (
        f"cross-check FAILED: True News.csv has {len(left):,} rows, Combined .csv has "
        f"{len(right):,} TRUE rows; {only_standalone:,} texts appear only in the "
        f"standalone file and {only_combined:,} only in Combined .csv"
    )


def check_fake_news_still_corrupt() -> tuple[bool, str]:
    """Assert `Fake News.csv` is still the corrupted file M1-3 documented.

    Deliberately inverted logic: this passes when the file is STILL broken. If the
    upstream repository ever republishes a fixed version, this check fails loudly —
    which is the point. A silent upstream "fix" would change the dataset under the
    project's feet and invalidate every committed result, and M1-3's whole rationale
    for reading `Combined .csv` would need revisiting.
    """
    from research.src.data.download import sha256_of

    if not AX_TO_GRIND_CORRUPT_EVIDENCE.exists():
        return False, f"provenance check skipped: {AX_TO_GRIND_CORRUPT_EVIDENCE.name} is missing"

    digest = sha256_of(AX_TO_GRIND_CORRUPT_EVIDENCE)
    payload = AX_TO_GRIND_CORRUPT_EVIDENCE.read_bytes()
    question_marks = payload.count(0x3F)

    if digest != CORRUPT_FAKE_NEWS_SHA256:
        return False, (
            f"Fake News.csv CHANGED upstream: sha256 {digest} != documented "
            f"{CORRUPT_FAKE_NEWS_SHA256}. DECISION_REGISTER.md M1-3 must be revisited "
            "before this pipeline is trusted."
        )
    if question_marks != CORRUPT_FAKE_NEWS_QUESTION_MARK_BYTES:
        return False, (
            f"Fake News.csv corruption signature changed: {question_marks:,} '?' bytes, "
            f"expected {CORRUPT_FAKE_NEWS_QUESTION_MARK_BYTES:,}"
        )
    return True, (
        f"provenance OK: Fake News.csv is still the documented corrupted file "
        f"({question_marks:,} literal '?' bytes, {question_marks / len(payload):.1%} of it)"
    )


LOADERS = {"ax_to_grind": load_ax_to_grind, "notri_fact": load_notri_fact}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=sorted(LOADERS),
        help=(
            "Validate a single dataset instead of both. Mirrors download.py's flag, "
            "and exists for the same reason: a milestone that legitimately needs only "
            "one corpus present should not fail on the absence of the other. "
            "Milestone 4 (Experiments C, D) trains and evaluates on Ax-to-Grind alone; "
            "Notri-Fact is the held-out cross-dataset test set and is not read until "
            "Milestone 5 (EXPERIMENT_PLAN.md step 4). Default remains both datasets, "
            "so nothing narrows unless it is asked for explicitly."
        ),
    )
    args = parser.parse_args(argv)

    requested = (args.only,) if args.only else tuple(LOADERS)
    failures: list[str] = []

    # Both M1-3 checks read Ax-to-Grind files only, so they run whenever it is in
    # scope and are skipped entirely when it is not.
    if "ax_to_grind" in requested:
        print("=== M1-3 source checks ===")
        ok, message = check_fake_news_still_corrupt()
        print(f"  {message}")
        if not ok:
            failures.append("fake-news provenance")

        ok, message = cross_check_true_news()
        print(f"  {message}")
        if not ok:
            failures.append("true-news cross-check")

    print("\n=== Loading and schema validation ===")
    print(f"  datasets in scope: {list(requested)}\n")
    for name in requested:
        frame, report = LOADERS[name]()
        print(report.render())
        try:
            validate_schema(frame, name)
            print(f"[{name}] schema             : OK")
        except ValidationError as exc:
            print(f"[{name}] schema             : FAILED — {exc}", file=sys.stderr)
            failures.append(f"{name} schema")
        print(check_documented_size(frame, name))
        print()

    if failures:
        print(f"FAILED checks: {failures}", file=sys.stderr)
        return 1
    print("All validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
