"""Metrics output-schema tests (`ROADMAP.md` Milestone 3 test requirement).

Asserts `EXPERIMENT_PLAN.md` Section 4's output contract on both synthetic records
and every metrics file actually committed to `research/results/metrics/`.

The contract matters beyond tidiness: the model card generator, the `/research`
page and the thesis tables all parse these files. A field silently disappearing
would surface as a broken thesis table long after the commit that caused it.
"""

from __future__ import annotations

import json

import pytest

from research.src.evaluation.metrics import (
    METRICS_DIR,
    REQUIRED_METRIC_FIELDS,
    REQUIRED_PER_CLASS_FIELDS,
    REQUIRED_TOP_LEVEL_FIELDS,
    build_metrics_record,
    compute_metrics,
    majority_class_baseline,
    validate_metrics_record,
)

LABELS = ["fake", "real"]


def _record(y_true: list[str], y_pred: list[str]) -> dict:
    return build_metrics_record(
        experiment_id="TEST",
        model="fixture",
        dataset="ax_to_grind",
        split="test",
        seed=42,
        y_true=y_true,
        y_pred=y_pred,
        labels=LABELS,
    )


# --- Contract on synthetic records ---------------------------------------------


def test_record_has_every_required_field() -> None:
    record = _record(["fake", "real", "fake", "real"], ["fake", "real", "real", "real"])
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        assert field in record, f"missing required top-level field {field}"


def test_record_has_every_required_metric() -> None:
    record = _record(["fake", "real"], ["fake", "real"])
    for field in REQUIRED_METRIC_FIELDS:
        assert field in record["metrics"]


def test_record_has_per_class_precision_recall() -> None:
    record = _record(["fake", "real", "fake"], ["fake", "real", "real"])
    for label in LABELS:
        assert label in record["per_class"]
        for field in REQUIRED_PER_CLASS_FIELDS:
            assert field in record["per_class"][label]


def test_confusion_matrix_is_square_and_sums_to_n() -> None:
    y_true = ["fake", "real", "fake", "real"]
    y_pred = ["fake", "fake", "real", "real"]
    record = _record(y_true, y_pred)
    matrix = record["confusion_matrix"]["matrix"]
    assert len(matrix) == len(LABELS)
    assert all(len(row) == len(LABELS) for row in matrix)
    assert sum(sum(row) for row in matrix) == len(y_true)


def test_confusion_matrix_label_order_is_fixed() -> None:
    """Row/column order must come from config, so axes never silently swap."""
    assert _record(["fake"], ["fake"])["confusion_matrix"]["labels"] == LABELS


def test_validate_accepts_a_well_formed_record() -> None:
    assert validate_metrics_record(_record(["fake", "real"], ["fake", "real"])) == []


# --- The validator must actually reject malformed records ----------------------


def test_validator_rejects_missing_metric() -> None:
    record = _record(["fake", "real"], ["fake", "real"])
    del record["metrics"]["macro_f1"]
    assert any("macro_f1" in problem for problem in validate_metrics_record(record))


def test_validator_rejects_out_of_range_metric() -> None:
    record = _record(["fake", "real"], ["fake", "real"])
    record["metrics"]["accuracy"] = 1.5
    assert any("outside [0,1]" in problem for problem in validate_metrics_record(record))


def test_validator_rejects_inconsistent_confusion_matrix() -> None:
    record = _record(["fake", "real"], ["fake", "real"])
    record["confusion_matrix"]["matrix"] = [[99, 0], [0, 0]]
    assert any("sum to n_examples" in problem for problem in validate_metrics_record(record))


# --- Metric correctness --------------------------------------------------------


def test_perfect_predictions_score_one() -> None:
    metrics = compute_metrics(["fake", "real"], ["fake", "real"], labels=LABELS)["metrics"]
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_prediction_collapse_is_detected() -> None:
    """The failure mode macro-F1 exists to expose (Haroon 2026: 99.7% one class)."""
    y_true = ["fake"] * 50 + ["real"] * 50
    collapse = compute_metrics(y_true, ["fake"] * 100, labels=LABELS)["prediction_collapse"]
    assert collapse["is_collapsed"] is True
    assert collapse["dominant_class_share"] == 1.0


def test_healthy_predictions_are_not_flagged_as_collapsed() -> None:
    y_true = ["fake"] * 50 + ["real"] * 50
    y_pred = ["fake"] * 48 + ["real"] * 52
    assert compute_metrics(y_true, y_pred, labels=LABELS)["prediction_collapse"][
        "is_collapsed"
    ] is False


def test_majority_class_baseline_breaks_ties_deterministically() -> None:
    """A tie must resolve the same way in every process.

    The Ax-to-Grind test split is exactly 687/687, so this is the real case, not a
    hypothetical. An earlier implementation used `max(set(...), key=count)`, which
    resolved ties by set iteration order and flipped between runs under string
    hash randomisation — every score stayed identical while the recorded
    `majority_class` changed, leaving a metrics file that could not be reproduced.
    """
    balanced = ["real"] * 10 + ["fake"] * 10
    first = majority_class_baseline(balanced, LABELS)["majority_class"]
    for ordering in (balanced, list(reversed(balanced)), sorted(balanced)):
        assert majority_class_baseline(ordering, LABELS)["majority_class"] == first


def test_majority_class_baseline_picks_the_actual_majority() -> None:
    """Tie-breaking must not override a genuine majority."""
    skewed = ["real"] * 3 + ["fake"] * 17
    assert majority_class_baseline(skewed, LABELS)["majority_class"] == "fake"


def test_majority_class_baseline_on_balanced_data_is_one_third() -> None:
    """A collapsed predictor scores 0.333 macro-F1 on balanced binary data.

    This is the floor ROADMAP.md's acceptance criterion measures against.
    """
    baseline = majority_class_baseline(["fake"] * 50 + ["real"] * 50, LABELS)
    assert baseline["macro_f1"] == pytest.approx(1 / 3, abs=0.01)
    assert baseline["accuracy"] == pytest.approx(0.5, abs=0.01)


# --- Every committed metrics file must conform ---------------------------------


def _committed_metrics_files() -> list:
    return sorted(METRICS_DIR.glob("*.json")) if METRICS_DIR.exists() else []


def test_metrics_directory_contains_results() -> None:
    """ROADMAP.md Milestone 3: A_*, B_*, H_* must exist with real numbers.

    H2_* joins them via EXPERIMENT_PLAN.md step 2b (DECISION_REGISTER.md M3-1),
    which places H2 in the REQUIRED set rather than treating it as optional.
    """
    files = _committed_metrics_files()
    if not files:
        pytest.skip("no metrics generated yet — run the experiment scripts")
    names = {path.name.split("_")[0] for path in files}
    for experiment in ("A", "B", "H", "H2"):
        assert experiment in names, f"no metrics file for experiment {experiment}"


# --- The H/H2 floor-ceiling pair (DECISION_REGISTER.md M3-1) -------------------


def _h2_files() -> list:
    return sorted(METRICS_DIR.glob("H2_*.json")) if METRICS_DIR.exists() else []


@pytest.mark.parametrize("path", _h2_files(), ids=lambda p: p.name)
def test_h2_records_the_floor_ceiling_pair(path) -> None:
    """H2's number is not interpretable alone, so its file must carry H's too."""
    record = json.loads(path.read_text(encoding="utf-8"))
    pair = record["run_metadata"]["length_shortcut_floor_ceiling"]
    for field in (
        "floor_H_linear_macro_f1",
        "ceiling_H2_nonlinear_macro_f1",
        "delta_H2_minus_H",
        "majority_class_macro_f1",
    ):
        assert field in pair, f"{path.name}: floor/ceiling pair missing {field}"


@pytest.mark.parametrize("path", _h2_files(), ids=lambda p: p.name)
def test_h2_floor_ceiling_pair_matches_the_committed_files(path) -> None:
    """The pair is copied from H's file at write time, so it can go stale if H is
    re-run afterwards. This is what catches that — two numbers in the repo
    disagreeing about the same experiment is exactly the failure `CLAUDE.md` rule
    2 exists to prevent."""
    record = json.loads(path.read_text(encoding="utf-8"))
    pair = record["run_metadata"]["length_shortcut_floor_ceiling"]

    assert pair["ceiling_H2_nonlinear_macro_f1"] == pytest.approx(
        record["metrics"]["macro_f1"], abs=1e-4
    )

    h_path = METRICS_DIR / pair["source_files"]["H"]
    if not h_path.exists():
        pytest.skip(f"{h_path.name} not committed — nothing to cross-check")
    h_macro_f1 = json.loads(h_path.read_text(encoding="utf-8"))["metrics"]["macro_f1"]
    assert pair["floor_H_linear_macro_f1"] == pytest.approx(h_macro_f1, abs=1e-4), (
        f"{path.name} records H={pair['floor_H_linear_macro_f1']} but "
        f"{h_path.name} says {h_macro_f1:.4f} — re-run H2 to refresh the pair"
    )
    assert pair["delta_H2_minus_H"] == pytest.approx(
        record["metrics"]["macro_f1"] - h_macro_f1, abs=1e-4
    )


@pytest.mark.parametrize("path", _h2_files(), ids=lambda p: p.name)
def test_h2_reports_the_configured_depth_not_a_sweep_probe(path) -> None:
    """Guards against the headline number silently becoming a diagnostic probe's.

    The depth sweep refits at other depths; if that ever leaked into the reported
    result, the committed macro-F1 would no longer match the config the file
    claims produced it.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    configured = record["config"]["model"]["experiment"]["params"]["max_depth"]
    sweep = record["run_metadata"]["depth_sweep_diagnostic"]

    assert sweep["reported_depth"] == configured
    assert record["run_metadata"]["tree_structure"]["max_depth_configured"] == configured
    assert sweep["macro_f1_by_depth"][f"max_depth_{configured}"] == pytest.approx(
        record["metrics"]["macro_f1"], abs=1e-4
    )


@pytest.mark.parametrize("path", _h2_files(), ids=lambda p: p.name)
def test_h2_never_touched_article_content(path) -> None:
    """The one claim H2 cannot afford to be wrong about."""
    metadata = json.loads(path.read_text(encoding="utf-8"))["run_metadata"]
    assert metadata["uses_text_content"] is False
    assert metadata["classifier"] == "decision_tree"


@pytest.mark.parametrize(
    "path", _committed_metrics_files(), ids=lambda p: p.name
)
def test_committed_metrics_file_conforms_to_contract(path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_metrics_record(record)
    assert not problems, f"{path.name} violates the output contract: {problems}"


@pytest.mark.parametrize(
    "path", _committed_metrics_files(), ids=lambda p: p.name
)
def test_committed_metrics_are_not_degenerate(path) -> None:
    """Acceptance criterion: meaningfully above the majority-class floor, and not
    a collapsed all-one-class predictor."""
    record = json.loads(path.read_text(encoding="utf-8"))
    macro_f1 = record["metrics"]["macro_f1"]
    floor = record["majority_class_baseline"]["macro_f1"]

    assert macro_f1 > floor + 0.05, (
        f"{path.name}: macro-F1 {macro_f1:.4f} is not meaningfully above the "
        f"majority-class floor {floor:.4f}"
    )
    assert not record["prediction_collapse"]["is_collapsed"], (
        f"{path.name}: degenerate all-one-class predictor"
    )
