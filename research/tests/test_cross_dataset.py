"""Tests for Experiment F — cross-dataset zero-shot transfer (`TESTING_STRATEGY.md`).

Two things are checked, following the patterns already established by Milestone 3's
tests: the committed F metrics files satisfy the same output contract every other
experiment writes (`test_metrics_schema.py`'s approach), and the classical retrain
path is deterministic (`test_feature_determinism.py`'s approach) — the same seed
must give the same model, or the "retrain rather than persist a pickle" design in
`run_cross_dataset.py` would silently produce a different model each run.

A third thing is asserted that matters more than either for RQ2's validity: that
these files really are **zero-shot**. If the target corpus ever leaked into
training, the headline cross-dataset number would be meaningless, and the failure
would be invisible in the metrics themselves.
"""

from __future__ import annotations

import json

import pytest

from research.src.data.split import load_split_frame
from research.src.evaluation.metrics import (
    METRICS_DIR,
    REQUIRED_METRIC_FIELDS,
    REQUIRED_TOP_LEVEL_FIELDS,
    validate_metrics_record,
)
from research.src.experiments.run_cross_dataset import (
    DIRECTIONS,
    load_config,
    run_classical_transfer,
)
from research.src.models.classical import build_pipeline, fit_predict


def _f_files() -> list:
    return sorted(METRICS_DIR.glob("F_*.json")) if METRICS_DIR.exists() else []


@pytest.fixture
def classical_config() -> dict:
    return load_config("model_classical.yaml")


# --------------------------------------------------------------------------
# output contract
# --------------------------------------------------------------------------


def test_cross_dataset_metrics_exist() -> None:
    assert _f_files(), "no F_*.json committed — run run_cross_dataset.py --models classical"


@pytest.mark.parametrize("path", _f_files(), ids=lambda p: p.name)
def test_f_file_satisfies_the_metrics_contract(path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        assert field in record, f"{path.name} missing {field}"
    for field in REQUIRED_METRIC_FIELDS:
        assert field in record["metrics"], f"{path.name} missing metrics.{field}"
    assert validate_metrics_record(record) == [], path.name


@pytest.mark.parametrize("path", _f_files(), ids=lambda p: p.name)
def test_f_file_is_evaluated_on_the_target_corpus(path) -> None:
    """The evaluated dataset must be the TARGET, never the source."""
    record = json.loads(path.read_text(encoding="utf-8"))
    direction = DIRECTIONS[record["experiment_id"]]
    assert record["dataset"] == direction["target_dataset"]
    assert record["split"] == direction["target_split"]


# --------------------------------------------------------------------------
# zero-shot integrity — the property RQ2 actually rests on
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", _f_files(), ids=lambda p: p.name)
def test_f_file_declares_zero_shot_transfer(path) -> None:
    transfer = json.loads(path.read_text(encoding="utf-8"))["run_metadata"]["transfer"]
    assert transfer["zero_shot"] is True
    assert transfer["target_used_for_training"] is False
    assert transfer["target_used_for_tuning"] is False


@pytest.mark.parametrize("path", _f_files(), ids=lambda p: p.name)
def test_source_and_target_are_different_corpora(path) -> None:
    """A 'cross-dataset' result computed within one corpus would be meaningless."""
    transfer = json.loads(path.read_text(encoding="utf-8"))["run_metadata"]["transfer"]
    assert transfer["source_dataset"] != transfer["target_dataset"]


def test_target_corpus_is_never_a_training_source_in_any_direction() -> None:
    """`data.yaml` marks Notri-Fact `train_on: false`; no direction may violate it."""
    data_config = load_config("data.yaml")
    for direction_id, direction in DIRECTIONS.items():
        source = direction["source_dataset"]
        train_on = data_config["datasets"][source].get("train_on", True)
        assert train_on is not False, (
            f"direction {direction_id} trains on {source}, which data.yaml marks "
            "train_on: false — this is the R7 / blueprint Part 9 constraint"
        )


def test_dedup_gate_is_recorded_in_the_metrics() -> None:
    """The zero-shot claim depends on the gate; the file must carry the evidence."""
    for path in _f_files():
        gate = json.loads(path.read_text(encoding="utf-8"))["run_metadata"]["transfer"]["dedup_gate"]
        assert gate["cross_dataset_near_duplicate_pairs"] == 0


def test_dedup_gate_actually_passed_in_the_audit_artifact() -> None:
    """Asserted against the audit output itself, not against what a run claimed."""
    import pathlib

    report = json.loads(
        (pathlib.Path(METRICS_DIR).parents[1] / "data/audit/duplication_report.json").read_text(
            encoding="utf-8"
        )
    )
    cross = [s for s in report["scans"] if s["scan"].startswith("cross:")]
    assert cross, "no cross-dataset scan in the duplication report"
    for scan in cross:
        assert scan["near_duplicate_pairs"] == 0, scan
        assert scan["exact_duplicate_rows"] == 0, scan


# --------------------------------------------------------------------------
# determinism of the classical retrain path
# --------------------------------------------------------------------------


def test_classical_retrain_is_deterministic(classical_config: dict) -> None:
    """Same seed -> same predictions.

    `run_cross_dataset.py` refits from config rather than loading a persisted
    model, which is only sound if refitting is reproducible.
    """
    direction = DIRECTIONS["F"]
    train = load_split_frame(direction["source_dataset"], direction["source_split"])
    # A slice keeps the test fast; determinism does not depend on corpus size.
    target = load_split_frame(direction["target_dataset"], direction["target_split"]).head(200)

    runs = []
    for _ in range(2):
        pipeline = build_pipeline(classical_config, "A")
        predictions, scores = fit_predict(
            pipeline,
            train_texts=train["text"].tolist(),
            train_labels=train["label"].tolist(),
            eval_texts=target["text"].tolist(),
            positive_label="fake",
        )
        runs.append((list(predictions), list(scores)))

    assert runs[0][0] == runs[1][0], "predictions differ between identical runs"
    assert runs[0][1] == pytest.approx(runs[1][1]), "scores differ between identical runs"


def test_classical_retrain_matches_the_committed_result(classical_config: dict) -> None:
    """Refitting reproduces the committed F number, so the file is regenerable."""
    committed = [p for p in _f_files() if "tfidf_logreg" in p.name]
    if not committed:
        pytest.skip("F logreg result not committed yet")
    expected = json.loads(committed[0].read_text(encoding="utf-8"))["metrics"]["macro_f1"]

    direction = DIRECTIONS["F"]
    train = load_split_frame(direction["source_dataset"], direction["source_split"])
    target = load_split_frame(direction["target_dataset"], direction["target_split"])

    from sklearn.metrics import f1_score

    pipeline = build_pipeline(classical_config, "A")
    predictions, _ = fit_predict(
        pipeline,
        train_texts=train["text"].tolist(),
        train_labels=train["label"].tolist(),
        eval_texts=target["text"].tolist(),
        positive_label="fake",
    )
    recomputed = f1_score(target["label"].tolist(), list(predictions), average="macro")
    assert recomputed == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------
# Experiment G is deliberately absent
# --------------------------------------------------------------------------


def test_experiment_g_is_not_silently_implemented() -> None:
    """G requires training on Notri-Fact, which the blueprint both requires
    (Part 11) and forbids (Part 9). Until that is resolved in DECISION_REGISTER.md,
    G must not exist — an accidental implementation would produce a headline RQ2
    number resting on an unresolved contradiction."""
    assert "G" not in DIRECTIONS
    assert not list(METRICS_DIR.glob("G_*.json")), (
        "G_*.json exists but Experiment G is unresolved — see run_cross_dataset.py"
    )


def test_run_classical_transfer_rejects_an_unknown_direction(classical_config: dict) -> None:
    with pytest.raises(KeyError):
        run_classical_transfer(
            "G",
            "A",
            data_config=load_config("data.yaml"),
            model_config=classical_config,
        )
