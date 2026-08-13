"""Schema and data-validation tests (`TESTING_STRATEGY.md`, "Data validation tests").

Real pytest assertions over the actual loaded datasets, not print statements in a
notebook. Tests needing the raw files skip when they are absent, for the same reason
as `test_raw_data_integrity.py`: `research/data/raw/` is gitignored, so a fresh clone
legitimately has no data until `download.py` runs. Tests over pure logic (label
normalisation, the known-issue handling) run unconditionally against fixtures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.src.data.validate import (
    AX_TO_GRIND_SOURCE,
    CANONICAL_LABELS,
    DOCUMENTED_SIZES,
    LABEL_FAKE,
    LABEL_REAL,
    NOTRI_FACT_SOURCE,
    SIZE_TOLERANCE,
    ValidationError,
    _normalise_label,
    check_fake_news_still_corrupt,
    cross_check_true_news,
    load_ax_to_grind,
    load_notri_fact,
    validate_schema,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _require(path) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} not present — run research/src/data/download.py")


@pytest.fixture(scope="module")
def ax_to_grind() -> pd.DataFrame:
    _require(AX_TO_GRIND_SOURCE)
    frame, _ = load_ax_to_grind()
    return frame


@pytest.fixture(scope="module")
def notri_fact() -> pd.DataFrame:
    _require(NOTRI_FACT_SOURCE)
    frame, _ = load_notri_fact()
    return frame


# --- Pure-logic tests (no data files needed) ----------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("TRUE", LABEL_REAL),
        ("FAKE", LABEL_FAKE),
        ("TRUE ", LABEL_REAL),  # 33 rows carry a trailing space
        ("FAKE ", LABEL_FAKE),  # 14 rows carry a trailing space
        ("Real", LABEL_REAL),  # Notri-Fact
        ("Unreal", LABEL_FAKE),  # Notri-Fact
        ("  real  ", LABEL_REAL),
        ("Label", None),  # the stray header row
        ("", None),
        (None, None),
    ],
)
def test_label_normalisation(raw: object, expected: str | None) -> None:
    assert _normalise_label(raw) == expected


def test_validate_schema_rejects_duplicate_row_ids() -> None:
    frame = pd.DataFrame(
        {"row_id": ["a", "a"], "text": ["x", "y"], "label": [LABEL_REAL, LABEL_FAKE]}
    )
    with pytest.raises(ValidationError, match="row_id is not unique"):
        validate_schema(frame, "fixture")


def test_validate_schema_rejects_blank_text() -> None:
    frame = pd.DataFrame(
        {"row_id": ["a", "b"], "text": ["x", "   "], "label": [LABEL_REAL, LABEL_FAKE]}
    )
    with pytest.raises(ValidationError, match="blank text"):
        validate_schema(frame, "fixture")


def test_validate_schema_rejects_a_third_label() -> None:
    frame = pd.DataFrame(
        {"row_id": ["a", "b", "c"], "text": ["x", "y", "z"], "label": ["real", "fake", "maybe"]}
    )
    with pytest.raises(ValidationError, match="expected exactly the labels"):
        validate_schema(frame, "fixture")


# --- Ax-to-Grind ---------------------------------------------------------------


def test_ax_to_grind_has_two_labels(ax_to_grind: pd.DataFrame) -> None:
    """Exactly two label values, no more (DATASET_PLAN.md Section 2 step 2)."""
    assert set(ax_to_grind["label"].unique()) == set(CANONICAL_LABELS)


def test_no_null_text_after_cleaning(ax_to_grind: pd.DataFrame) -> None:
    """No null or whitespace-only article text survives loading.

    The 22 blank rows and the stray header row in the raw file are dropped by the
    loader; this asserts none leaked through.
    """
    assert int(ax_to_grind["text"].isna().sum()) == 0
    assert int(ax_to_grind["text"].map(str).str.strip().eq("").sum()) == 0


def test_ax_to_grind_row_count_matches_documented_size(ax_to_grind: pd.DataFrame) -> None:
    expected = DOCUMENTED_SIZES["ax_to_grind"]
    assert abs(len(ax_to_grind) - expected) <= expected * SIZE_TOLERANCE, (
        f"{len(ax_to_grind):,} rows vs documented {expected:,}"
    )


def test_ax_to_grind_row_ids_unique(ax_to_grind: pd.DataFrame) -> None:
    """Split index files reference rows by id, so ids must identify a row."""
    assert not ax_to_grind["row_id"].duplicated().any()


def test_ax_to_grind_schema(ax_to_grind: pd.DataFrame) -> None:
    validate_schema(ax_to_grind, "ax_to_grind")


def test_ax_to_grind_stray_header_row_removed(ax_to_grind: pd.DataFrame) -> None:
    """The literal 'News Items' header string must not survive as an article."""
    assert not (ax_to_grind["text"].map(str).str.strip() == "News Items").any()


def test_ax_to_grind_is_urdu_script(ax_to_grind: pd.DataFrame) -> None:
    """Guards against silently loading the corrupted Fake News.csv by mistake.

    If the loader were ever repointed at the corrupted file, its text would be
    ~73% literal '?' with essentially no Arabic-script characters, and this fails.
    """
    sample = ax_to_grind["text"].map(str).head(2000)
    arabic = sample.map(lambda s: sum(1 for ch in s if "؀" <= ch <= "ۿ")).sum()
    total = sample.map(len).sum()
    assert arabic / total > 0.5, (
        f"only {arabic / total:.1%} of sampled characters are Arabic-script — "
        "is the loader reading the corrupted Fake News.csv?"
    )


# --- Notri-Fact ----------------------------------------------------------------


def test_notri_fact_has_two_labels(notri_fact: pd.DataFrame) -> None:
    assert set(notri_fact["label"].unique()) == set(CANONICAL_LABELS)


def test_notri_fact_no_null_text(notri_fact: pd.DataFrame) -> None:
    assert int(notri_fact["text"].isna().sum()) == 0
    assert int(notri_fact["text"].map(str).str.strip().eq("").sum()) == 0


def test_notri_fact_row_count_matches_documented_size(notri_fact: pd.DataFrame) -> None:
    expected = DOCUMENTED_SIZES["notri_fact"]
    assert abs(len(notri_fact) - expected) <= expected * SIZE_TOLERANCE, (
        f"{len(notri_fact):,} rows vs documented {expected:,}"
    )


def test_notri_fact_schema(notri_fact: pd.DataFrame) -> None:
    validate_schema(notri_fact, "notri_fact")


def test_notri_fact_retains_category_for_audit(notri_fact: pd.DataFrame) -> None:
    """audit.py needs Category for the domain cross-tab and domain-shift check."""
    assert "Category" in notri_fact.columns
    assert int(notri_fact["Category"].isna().sum()) == 0


# --- M1-3 source guarantees ----------------------------------------------------


def test_fake_news_csv_is_still_the_documented_corrupt_file() -> None:
    """Inverted on purpose: passes while the upstream file is still broken.

    A failure here means upstream republished the file, which would require
    revisiting DECISION_REGISTER.md M1-3 before trusting the pipeline.
    """
    from research.src.data.validate import AX_TO_GRIND_CORRUPT_EVIDENCE

    _require(AX_TO_GRIND_CORRUPT_EVIDENCE)
    ok, message = check_fake_news_still_corrupt()
    assert ok, message


def test_true_news_cross_check_passes() -> None:
    """True News.csv's rows must equal Combined .csv's TRUE rows as a multiset."""
    from research.src.data.validate import AX_TO_GRIND_TRUE_CROSSCHECK

    _require(AX_TO_GRIND_TRUE_CROSSCHECK)
    _require(AX_TO_GRIND_SOURCE)
    ok, message = cross_check_true_news()
    assert ok, message
