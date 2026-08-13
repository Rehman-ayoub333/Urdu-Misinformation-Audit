"""Deduplication tests, including the cross-dataset go/no-go gate.

`ROADMAP.md` Milestone 2 makes zero cross-dataset duplication a **hard gate**: any
overlap means an "unseen" Notri-Fact article was in fact trained on, which silently
invalidates RQ2's zero-shot claim.

A gate that passes is only meaningful if it is capable of failing. The real-data
assertion is therefore paired with **positive controls** that plant known duplicates
and require the scan to catch them. Without those, a detector that always returned
"no duplicates found" would pass the gate and nobody would know.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.src.data.clean import clean_text
from research.src.data.dedup import (
    NEAR_DUPLICATE_THRESHOLD,
    _jaccard,
    _shingles,
    find_cross_dataset_duplicates,
    find_within_dataset_duplicates,
    minhash_signature,
)
from research.src.data.validate import (
    AX_TO_GRIND_SOURCE,
    NOTRI_FACT_SOURCE,
    load_ax_to_grind,
    load_notri_fact,
)


def _require_data() -> None:
    for path in (AX_TO_GRIND_SOURCE, NOTRI_FACT_SOURCE):
        if not path.exists():
            pytest.skip(f"{path.name} not present — run research/src/data/download.py")


@pytest.fixture(scope="module")
def datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_data()
    axg, _ = load_ax_to_grind()
    ntf, _ = load_notri_fact()
    return axg, ntf


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [r[0] for r in rows],
            "text": [r[1] for r in rows],
            "label": ["real"] * len(rows),
        }
    )


# --- MinHash unit behaviour ----------------------------------------------------


def test_shingles_of_short_text_is_the_text_itself() -> None:
    assert _shingles("abc") == {"abc"}


def test_shingles_are_character_ngrams() -> None:
    assert _shingles("abcdefg", size=5) == {"abcde", "bcdef", "cdefg"}


def test_identical_text_has_identical_signature() -> None:
    text = "پاکستان کی معاشی صورتحال بہتر ہو رہی ہے۔ یہ ایک لمبا جملہ ہے۔"
    assert minhash_signature(text) == minhash_signature(text)


def test_minhash_is_deterministic_across_calls() -> None:
    """Seeded coefficients: a rerun must reproduce the committed dedup decisions."""
    a = minhash_signature("some reasonably long document text for hashing purposes")
    b = minhash_signature("some reasonably long document text for hashing purposes")
    assert a == b


def test_jaccard_of_identical_signatures_is_one() -> None:
    sig = minhash_signature("a moderately long piece of text to shingle properly")
    assert _jaccard(sig, sig) == 1.0


def test_jaccard_of_unrelated_text_is_low() -> None:
    a = minhash_signature("cricket match report from lahore stadium yesterday evening")
    b = minhash_signature("economic policy announcement regarding interest rates today")
    assert _jaccard(a, b) < NEAR_DUPLICATE_THRESHOLD


# --- Within-dataset detection --------------------------------------------------


def test_within_scan_detects_exact_duplicates() -> None:
    text = "یہ ایک مکمل خبر ہے جو دو مرتبہ موجود ہے اور اسے پکڑا جانا چاہیے۔"
    frame = _frame([("a-1", text), ("a-2", text), ("a-3", "کچھ اور بالکل مختلف خبر یہاں ہے۔")])
    report = find_within_dataset_duplicates(frame, "fixture")
    assert report.exact_duplicate_rows == 1
    assert report.rows_to_remove == ["a-2"]


def test_within_scan_keeps_first_occurrence() -> None:
    """Deterministic survivor choice, so reruns produce the same surviving set."""
    text = "ایک ہی خبر تین مرتبہ دہرائی گئی ہے تاکہ جانچ کی جا سکے۔"
    frame = _frame([("a-1", text), ("a-2", text), ("a-3", text)])
    report = find_within_dataset_duplicates(frame, "fixture")
    assert "a-1" not in report.rows_to_remove
    assert set(report.rows_to_remove) == {"a-2", "a-3"}


# --- THE GATE: positive controls -----------------------------------------------


def test_cross_scan_detects_a_planted_exact_duplicate(
    datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """POSITIVE CONTROL. Plant a real Ax-to-Grind article into Notri-Fact.

    If this does not fire, the clean bill of health on the real data means nothing.
    """
    axg, ntf = datasets
    stolen = axg["text"].iloc[0]

    contaminated = pd.concat(
        [ntf[["row_id", "text", "label"]].head(300), _frame([("ntf-PLANTED", stolen)])],
        ignore_index=True,
    )
    report = find_cross_dataset_duplicates(axg.head(2000), contaminated)

    assert "ntf-PLANTED" in report.rows_to_remove, (
        "the cross-dataset scan failed to detect a verbatim planted duplicate — "
        "the gate is vacuous and its passing result cannot be trusted"
    )


def test_cross_scan_detects_a_planted_near_duplicate(
    datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """POSITIVE CONTROL for NEAR duplicates, which is the harder case.

    Exact-hash matching alone would miss a republished article carrying a changed
    headline or an appended byline, so the planted text is modified at both ends.

    A *long* article is chosen deliberately. Ax-to-Grind's median cleaned length is
    only 143 characters, and on a 67-character item a 34-character byline is a ~35%
    content change whose true Jaccard is ~0.65 — genuinely not a near-duplicate at
    the 0.8 threshold. Using a short item here would assert the detector should fire
    on something that is not a duplicate. Real republication is a small edit to a
    full article, which is what this models.
    """
    axg, ntf = datasets
    # Select the source article from the SAME subset that is scanned — picking the
    # global longest would land outside head(2000) and the scan would correctly
    # find nothing, failing the test for a reason unrelated to detection.
    haystack = axg.head(2000)
    lengths = haystack["text"].map(str).map(len)
    stolen = str(haystack.loc[lengths.idxmax(), "text"])
    modified = "تازہ ترین خبر: " + stolen + " (رپورٹ: نامہ نگار)"

    contaminated = pd.concat(
        [ntf[["row_id", "text", "label"]].head(300), _frame([("ntf-NEARDUP", modified)])],
        ignore_index=True,
    )
    report = find_cross_dataset_duplicates(haystack, contaminated)

    assert "ntf-NEARDUP" in report.rows_to_remove, (
        "the cross-dataset scan missed a near-duplicate with modified head and tail"
    )


def test_minhash_estimate_tracks_true_jaccard() -> None:
    """The estimator must be accurate, since the gate's verdict rests on it.

    Pins MinHash's estimate against the exact Jaccard computed from the shingle
    sets. This is what justifies trusting a "zero duplicates" result: the scan does
    not miss pairs because of a broken estimator, it reports low similarity because
    the similarity genuinely is low.
    """
    base = "پاکستان کی معاشی صورتحال بہتر ہو رہی ہے۔ " * 20
    for suffix in ("", "اضافی متن یہاں۔", "بالکل مختلف موضوع پر ایک اور خبر۔ " * 5):
        other = base + suffix
        exact = len(_shingles(base) & _shingles(other)) / len(_shingles(base) | _shingles(other))
        estimate = _jaccard(minhash_signature(base), minhash_signature(other))
        assert abs(estimate - exact) < 0.1, (
            f"MinHash estimate {estimate:.3f} drifted from true Jaccard {exact:.3f}"
        )


def test_cross_scan_only_ever_removes_from_the_test_side(
    datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Overlaps are removed from Notri-Fact, never Ax-to-Grind.

    MASTER_PROJECT_BLUEPRINT.md Part 9: removing from the training side would
    shrink the training set to fix a test-set problem.
    """
    axg, ntf = datasets
    stolen = axg["text"].iloc[0]
    contaminated = pd.concat(
        [ntf[["row_id", "text", "label"]].head(200), _frame([("ntf-PLANTED", stolen)])],
        ignore_index=True,
    )
    report = find_cross_dataset_duplicates(axg.head(1000), contaminated)

    assert all(row_id.startswith("ntf-") for row_id in report.rows_to_remove), (
        f"scan flagged non-Notri-Fact rows: "
        f"{[r for r in report.rows_to_remove if not r.startswith('ntf-')]}"
    )


# --- THE GATE: the real assertion ----------------------------------------------


def test_zero_cross_dataset_duplicates(datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """HARD GATE (ROADMAP.md Milestone 2 acceptance criterion).

    Zero articles may be shared between Ax-to-Grind and Notri-Fact after Tier-2
    cleaning. Any overlap invalidates RQ2's zero-shot claim, so this blocks
    Milestone 3 rather than merely warning.

    Trustworthy only because the positive controls above prove this same scan
    detects planted exact and near duplicates.
    """
    axg, ntf = datasets
    axg_clean = axg["text"].map(clean_text)
    ntf_clean = ntf["text"].map(clean_text)

    report = find_cross_dataset_duplicates(
        axg, ntf, left_texts=axg_clean, right_texts=ntf_clean
    )

    assert report.exact_duplicate_rows == 0, (
        f"{report.exact_duplicate_rows} exact cross-dataset duplicates found"
    )
    assert report.near_duplicate_pairs == 0, (
        f"{report.near_duplicate_pairs} near-duplicate cross-dataset pairs found "
        f"(Jaccard >= {NEAR_DUPLICATE_THRESHOLD}): {report.sample_pairs[:3]}"
    )
    assert report.total_flagged == 0
