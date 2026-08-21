"""The metrics contract every experiment writes.

Implements `EXPERIMENT_PLAN.md` Section 4's output contract and Section 5's metric
set. Every experiment in the project — classical, transformer, cross-dataset —
goes through `compute_metrics` and `write_metrics`, so a metrics file from
Milestone 3 has the same shape as one from Milestone 5 and downstream consumers
(the `/research` page, the model card generator, the thesis tables) need exactly
one parser.

Primary metric is **macro-F1** (`DECISION_REGISTER.md` R5), chosen because it
penalises the prediction-collapse failure mode central to this thesis — a model
that predicts one class for everything scores near-chance macro-F1 while accuracy
can still look respectable.

`prediction_collapse` is computed and stored on every run for that exact reason:
Haroon (2026) reports a model predicting "fake" for 99.7% of inputs, and a metrics
file that recorded only aggregate scores would hide it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

_ROOT = Path(__file__).resolve().parents[3]
METRICS_DIR = _ROOT / "research" / "results" / "metrics"
MANIFEST_PATH = _ROOT / "research" / "data" / "raw" / "MANIFEST.sha256"

# Every metrics file must carry these. Asserted by the schema test in
# research/tests/test_metrics_schema.py so the contract cannot silently drift.
REQUIRED_TOP_LEVEL_FIELDS = (
    "experiment_id",
    "model",
    "dataset",
    "split",
    "seed",
    "n_examples",
    "metrics",
    "confusion_matrix",
    "per_class",
    "run_metadata",
)

REQUIRED_METRIC_FIELDS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
)

REQUIRED_PER_CLASS_FIELDS = ("precision", "recall", "f1", "support")


def _git_commit() -> str | None:
    """Record which commit produced a result. Best-effort; never fails a run."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _dataset_version() -> str | None:
    """The SHA256 manifest IS the dataset version (REPRODUCIBILITY.md Section 3)."""
    if not MANIFEST_PATH.exists():
        return None
    import hashlib

    return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()[:16]


def run_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Environment captured automatically, never hand-typed (REPRODUCIBILITY.md S5)."""
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_manifest_sha256_prefix": _dataset_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    if extra:
        meta.update(extra)
    return meta


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str],
    scores: Sequence[float] | None = None,
    positive_label: str | None = None,
) -> dict[str, Any]:
    """Compute the full required metric set.

    Args:
        y_true / y_pred: string labels ("fake" / "real").
        labels: fixed label order, so confusion-matrix axes never drift between
            runs. Comes from data.yaml's `labels.order`.
        scores: continuous scores for the positive class — `predict_proba[:, 1]`
            or `decision_function`. ROC-AUC and PR-AUC are computed only when
            supplied, since LinearSVC has no probabilities.
        positive_label: which label counts as positive for ROC/PR.

    Returns:
        A dict matching the structure the schema test asserts.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = list(labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    predicted_counts = {label: int(y_pred.count(label)) for label in labels}
    n = len(y_pred)
    dominant_share = max(predicted_counts.values()) / n if n else 0.0

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    if scores is not None and positive_label is not None:
        binary_truth = [1 if label == positive_label else 0 for label in y_true]
        # Guard: AUC is undefined when only one class is present.
        if len(set(binary_truth)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(binary_truth, list(scores)))
            metrics["pr_auc"] = float(average_precision_score(binary_truth, list(scores)))
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    return {
        "metrics": metrics,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.tolist(),
            "note": "rows = true label, columns = predicted label, in `labels` order",
        },
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "prediction_collapse": {
            "predicted_counts": predicted_counts,
            "dominant_class_share": round(dominant_share, 4),
            # Haroon (2026) reports 99.7% single-class prediction; 0.95 flags the
            # same failure mode early rather than leaving it to be noticed by eye.
            "is_collapsed": bool(dominant_share >= 0.95),
        },
    }


def majority_class_baseline(y_true: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    """Scores for always predicting the most frequent class.

    Recorded alongside every result so "meaningfully above chance" is a number in
    the file rather than a judgement made later from memory. This is the exact
    comparison ROADMAP.md's Milestone 3 acceptance criterion asks for.
    """
    y_true = list(y_true)
    # Tie-break deterministically. The splits are near-exactly balanced (the
    # Ax-to-Grind test split is 687/687), so this IS a tie in practice, and
    # `max(set(...), key=count)` would resolve it by set iteration order — which
    # varies between processes under string hash randomisation. That made the
    # recorded `majority_class` flip between runs while every score stayed
    # identical, leaving a metrics file that could not be reproduced byte-for-byte.
    # Highest count first, then alphabetical.
    counts = {label: y_true.count(label) for label in sorted(set(y_true))}
    majority = min(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    y_pred = [majority] * len(y_true)
    return {
        "majority_class": majority,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def build_metrics_record(
    *,
    experiment_id: str,
    model: str,
    dataset: str,
    split: str,
    seed: int | None,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    scores: Sequence[float] | None = None,
    positive_label: str | None = None,
    config: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a complete, contract-conforming metrics record."""
    computed = compute_metrics(
        y_true, y_pred, labels=labels, scores=scores, positive_label=positive_label
    )
    record: dict[str, Any] = {
        "experiment_id": experiment_id,
        "model": model,
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "n_examples": len(list(y_true)),
        **computed,
        "majority_class_baseline": majority_class_baseline(y_true, labels),
        "run_metadata": run_metadata(extra_metadata),
    }
    if config is not None:
        record["config"] = config
    return record


def write_metrics(record: dict[str, Any], filename: str, destination: Path | None = None) -> Path:
    """Write a metrics record to research/results/metrics/<filename>."""
    target = (destination or METRICS_DIR) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


# --- Per-example predictions (REPRODUCIBILITY.md Section 6) ---------------------
#
# Section 6 requires the FULL per-example predictions be saved "for the test and
# cross-dataset test sets specifically ... so any derived statistic can be
# recomputed without rerunning inference". Only aggregates were ever written, which
# is what blocked Milestone 5's error-analysis sample (DECISION_REGISTER.md M5-2).
#
# The qualifying splits live here rather than in each runner so a new experiment
# cannot quietly skip them: "test" is the in-domain held-out split, "holdout" is
# Notri-Fact's, which is the cross-dataset test set. Validation splits are
# deliberately excluded — Section 6 scopes the requirement to test sets, and val
# predictions are used for model selection, not for reported findings.
PREDICTION_SPLITS = ("test", "holdout")


def should_persist_predictions(split: str) -> bool:
    return split in PREDICTION_SPLITS


def predictions_filename(metrics_filename: str) -> str:
    """`X.json` -> `X.predictions.jsonl`, so the pair is obvious on disk."""
    stem = metrics_filename.removesuffix(".json")
    return f"{stem}.predictions.jsonl"


def write_predictions(
    *,
    metrics_filename: str,
    split: str,
    row_ids: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores: Sequence[float] | None = None,
    destination: Path | None = None,
) -> Path | None:
    """Persist row-level predictions beside a metrics file. No-op for other splits.

    JSONL rather than CSV or parquet: one self-describing JSON object per line, so
    it diffs readably in git, streams without loading the whole file, and needs no
    dependency beyond the standard library (parquet would mean adding pyarrow,
    which `CLAUDE.md` rule 16 does not permit without a stated need).

    Scores are written at FULL precision, not rounded — the point of Section 6 is
    that a different metric can be recomputed later, and rounding would silently
    cap how faithfully that can be done.

    `score_positive` is P(positive class), where the positive label is recorded in
    the sibling metrics JSON at `config.data.labels.positive` — it is not repeated
    on every row.
    """
    if not should_persist_predictions(split):
        return None

    row_ids, y_true, y_pred = list(row_ids), list(y_true), list(y_pred)
    if not (len(row_ids) == len(y_true) == len(y_pred)):
        raise ValueError(
            f"predictions length mismatch for {metrics_filename}: "
            f"{len(row_ids)} row_ids, {len(y_true)} y_true, {len(y_pred)} y_pred"
        )
    score_list = list(scores) if scores is not None else None
    if score_list is not None and len(score_list) != len(row_ids):
        raise ValueError(
            f"predictions length mismatch for {metrics_filename}: "
            f"{len(score_list)} scores vs {len(row_ids)} rows"
        )

    target = (destination or METRICS_DIR) / predictions_filename(metrics_filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row_id in enumerate(row_ids):
            row: dict[str, Any] = {
                # Traceable back to the committed split index, so the source text is
                # recoverable via research/src/data/split.py without guessing.
                "row_id": row_id,
                "y_true": y_true[index],
                "y_pred": y_pred[index],
            }
            if score_list is not None:
                row["score_positive"] = float(score_list[index])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


# --- Reading and annotating already-written results -----------------------------
#
# Ablation experiments (Q, I) are defined by their delta against an uncapped or
# unablated baseline, so each one has to read a committed counterpart's headline
# number and record its own delta. Both live here rather than in each runner so
# there is one implementation of "read a result off disk" and one of "add to a
# record without disturbing what metrics.py computed".


def committed_macro_f1(filename: str, source: Path | None = None) -> float | None:
    """Read a committed result's macro-F1, or `None` if it is not on disk.

    Deliberately tolerant: an experiment whose baseline has not been run yet must
    still produce a valid result, recording `null` rather than failing or — far
    worse — inventing a plausible number (`CLAUDE.md` rule 2).
    """
    path = (source or METRICS_DIR) / filename
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["metrics"]["macro_f1"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def annotate_metrics_file(
    path: Path, annotate: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    """Re-open a written metrics file, let the caller add metadata, rewrite it.

    For the facts that genuinely do not exist when `build_metrics_record` runs: a
    delta needs the record's own macro-F1, and a checkpoint's revision SHA is only
    assigned once the push completes, which cannot precede training.

    The callback is expected to touch `run_metadata` only. The record is
    re-validated afterwards, so an annotation that broke the contract cannot
    survive to disk.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    annotate(record)

    problems = validate_metrics_record(record)
    if problems:
        raise RuntimeError(
            f"{path.name} violates the metrics contract after annotation: {problems}"
        )

    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def validate_metrics_record(record: dict[str, Any]) -> list[str]:
    """Return a list of contract violations; empty means the record conforms.

    Used by the schema test and by the runners themselves, so a malformed record
    is caught at write time rather than discovered when the thesis tables are built.
    """
    problems: list[str] = []

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in record:
            problems.append(f"missing top-level field: {field}")

    metrics = record.get("metrics", {})
    for field in REQUIRED_METRIC_FIELDS:
        if field not in metrics:
            problems.append(f"missing metric: {field}")
        elif not isinstance(metrics[field], (int, float)):
            problems.append(f"metric {field} is not numeric: {metrics[field]!r}")
        elif not 0.0 <= float(metrics[field]) <= 1.0:
            problems.append(f"metric {field} outside [0,1]: {metrics[field]!r}")

    per_class = record.get("per_class", {})
    if not per_class:
        problems.append("per_class is empty")
    for label, values in per_class.items():
        for field in REQUIRED_PER_CLASS_FIELDS:
            if field not in values:
                problems.append(f"per_class[{label}] missing {field}")

    cm = record.get("confusion_matrix", {})
    if "matrix" not in cm or "labels" not in cm:
        problems.append("confusion_matrix must carry both `matrix` and `labels`")
    else:
        size = len(cm["labels"])
        if len(cm["matrix"]) != size or any(len(row) != size for row in cm["matrix"]):
            problems.append("confusion_matrix is not square in `labels`")
        elif sum(sum(row) for row in cm["matrix"]) != record.get("n_examples"):
            problems.append("confusion_matrix entries do not sum to n_examples")

    return problems
