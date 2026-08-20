"""Tests for per-example prediction persistence (REPRODUCIBILITY.md Section 6).

Section 6 requires the full per-example predictions be saved for test and
cross-dataset-test splits "so any derived statistic can be recomputed without
rerunning inference". `build_metrics_record` never wrote them, which is what
blocked Milestone 5's error-analysis sample (`DECISION_REGISTER.md` M5-2).

The properties worth protecting are that the rule fires on exactly the right
splits (a val file would inflate the committed set and is out of scope; a missing
test file re-creates the gap), that rows stay aligned with the metrics they
accompany, and — the point of the whole exercise — that the aggregate metrics can
actually be recomputed from the persisted rows.
"""

from __future__ import annotations

import json

import pytest

from research.src.evaluation.metrics import (
    METRICS_DIR,
    PREDICTION_SPLITS,
    build_metrics_record,
    predictions_filename,
    should_persist_predictions,
    write_predictions,
)

ROWS = ["r1", "r2", "r3", "r4"]
Y_TRUE = ["fake", "real", "fake", "real"]
Y_PRED = ["fake", "real", "real", "real"]
SCORES = [0.91, 0.12, 0.44, 0.20]


def _write(tmp_path, split="test", **kwargs):
    return write_predictions(
        metrics_filename=kwargs.pop("metrics_filename", "X_m_d_test.json"),
        split=split,
        row_ids=kwargs.pop("row_ids", ROWS),
        y_true=kwargs.pop("y_true", Y_TRUE),
        y_pred=kwargs.pop("y_pred", Y_PRED),
        scores=kwargs.pop("scores", SCORES),
        destination=tmp_path,
    )


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------
# which splits qualify
# --------------------------------------------------------------------------


def test_test_and_holdout_qualify() -> None:
    """`test` is the in-domain held-out split; `holdout` is Notri-Fact's, which is
    the cross-dataset test set. Both are what Section 6 names."""
    assert set(PREDICTION_SPLITS) == {"test", "holdout"}
    assert should_persist_predictions("test")
    assert should_persist_predictions("holdout")


def test_validation_split_does_not_qualify() -> None:
    """Section 6 scopes the requirement to test sets; val drives model selection."""
    assert not should_persist_predictions("val")


def test_write_is_a_no_op_for_val(tmp_path) -> None:
    assert _write(tmp_path, split="val") is None
    assert list(tmp_path.glob("*.jsonl")) == []


def test_write_produces_a_file_for_test(tmp_path) -> None:
    path = _write(tmp_path, split="test")
    assert path is not None and path.exists()


# --------------------------------------------------------------------------
# file shape
# --------------------------------------------------------------------------


def test_filename_is_a_sibling_of_the_metrics_file() -> None:
    assert predictions_filename("A_tfidf_logreg_ax_to_grind_test.json") == (
        "A_tfidf_logreg_ax_to_grind_test.predictions.jsonl"
    )


def test_one_json_object_per_row(tmp_path) -> None:
    rows = _read(_write(tmp_path))
    assert len(rows) == len(ROWS)
    assert [r["row_id"] for r in rows] == ROWS


def test_required_fields_present(tmp_path) -> None:
    """row_id is what makes the source text recoverable from the split index."""
    for row in _read(_write(tmp_path)):
        for field in ("row_id", "y_true", "y_pred", "score_positive"):
            assert field in row


def test_scores_are_not_rounded(tmp_path) -> None:
    """Rounding would silently cap how faithfully a derived metric can be recomputed."""
    precise = [0.123456789012345, 0.9, 0.5, 0.25]
    rows = _read(_write(tmp_path, scores=precise))
    assert rows[0]["score_positive"] == precise[0]


def test_scores_may_be_omitted(tmp_path) -> None:
    """LinearSVC has no probabilities; the file must still be written."""
    rows = _read(_write(tmp_path, scores=None))
    assert all("score_positive" not in r for r in rows)
    assert [r["y_pred"] for r in rows] == Y_PRED


# --------------------------------------------------------------------------
# alignment — a silent mismatch would corrupt every downstream analysis
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"row_ids": ROWS[:3]},
        {"y_true": Y_TRUE[:3]},
        {"y_pred": Y_PRED[:3]},
        {"scores": SCORES[:3]},
    ],
    ids=["short_row_ids", "short_y_true", "short_y_pred", "short_scores"],
)
def test_length_mismatch_raises(tmp_path, kwargs) -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        _write(tmp_path, **kwargs)


# --------------------------------------------------------------------------
# the actual point: metrics recomputable from the persisted rows
# --------------------------------------------------------------------------


def test_aggregates_are_recomputable_from_persisted_rows(tmp_path) -> None:
    """Section 6's stated purpose, asserted rather than assumed."""
    rows = _read(_write(tmp_path))
    record = build_metrics_record(
        experiment_id="X", model="m", dataset="d", split="test", seed=42,
        y_true=[r["y_true"] for r in rows],
        y_pred=[r["y_pred"] for r in rows],
        labels=["fake", "real"],
        scores=[r["score_positive"] for r in rows],
        positive_label="fake",
    )
    direct = build_metrics_record(
        experiment_id="X", model="m", dataset="d", split="test", seed=42,
        y_true=Y_TRUE, y_pred=Y_PRED, labels=["fake", "real"],
        scores=SCORES, positive_label="fake",
    )
    assert record["metrics"] == direct["metrics"]
    assert record["confusion_matrix"] == direct["confusion_matrix"]


# --------------------------------------------------------------------------
# the committed backfill
# --------------------------------------------------------------------------


def _committed_test_metrics() -> list:
    if not METRICS_DIR.exists():
        return []
    return [
        p for p in sorted(METRICS_DIR.glob("*.json"))
        if should_persist_predictions(json.loads(p.read_text(encoding="utf-8"))["split"])
    ]


@pytest.mark.parametrize("path", _committed_test_metrics(), ids=lambda p: p.name)
def test_committed_predictions_match_their_metrics_file(path) -> None:
    """Where a predictions file exists, it must agree with its metrics sibling.

    Not every test-split metrics file has one yet: C, D and F's transformer half
    are pending the next GPU session (M5-2), since backfilling them requires
    loading the checkpoints. Those are skipped rather than failed, so the gap is
    visible in the test report instead of being silently tolerated forever.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    sibling = path.parent / predictions_filename(path.name)
    if not sibling.exists():
        pytest.skip(f"{path.name}: predictions pending GPU backfill (M5-2)")

    rows = [json.loads(line) for line in sibling.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == record["n_examples"], f"{sibling.name}: row count != n_examples"

    labels = set(record["confusion_matrix"]["labels"])
    assert {r["y_true"] for r in rows} <= labels
    assert {r["y_pred"] for r in rows} <= labels
    assert len({r["row_id"] for r in rows}) == len(rows), "duplicate row_id"

    # Predicted-class counts must reproduce the committed collapse figures exactly.
    counts = {label: 0 for label in labels}
    for row in rows:
        counts[row["y_pred"]] += 1
    assert counts == record["prediction_collapse"]["predicted_counts"]
