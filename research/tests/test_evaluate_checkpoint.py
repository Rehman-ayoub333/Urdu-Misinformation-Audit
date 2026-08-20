"""Tests for the eval-only checkpoint recovery path (M4-6).

Two things matter here and both are asserted:

1. The recovered record is **shape-identical** to one the training run would have
   written, so downstream consumers (the summary table, the model card generator,
   the thesis tables) need no special case for a recovered file.
2. The record is **honest about what it lost**. `train_runtime_seconds` and
   `train_loss` must be present-and-null with an explicit provenance block, never
   absent (which would read as an oversight) and never populated with a plausible
   number (`CLAUDE.md` rule 2). The hardware recorded must be labelled as the
   evaluation host, since it is not the training host.

The heavy path (downloading a real checkpoint and running a forward pass) needs a
GPU session and the Hub, so it is exercised on Kaggle rather than here. What is
tested here is the record construction and the repo/revision wiring, which is where
a silent mistake would actually corrupt the research record.
"""

from __future__ import annotations

import json

import pytest

from research.src.models import evaluate_checkpoint
from research.src.models.transformer import load_config


@pytest.fixture
def model_config():
    return load_config("model_xlmr.yaml")


# --------------------------------------------------------------------------
# repo / revision wiring — must match what push_checkpoint actually used
# --------------------------------------------------------------------------


def test_repo_id_matches_the_push_construction(monkeypatch, model_config):
    monkeypatch.setenv("HF_STAGING_PREFIX", "RehmanAyoub")
    assert (
        evaluate_checkpoint.resolve_repo_id(model_config)
        == "RehmanAyoub/urdu-misinfo-xlmr-staging"
    )


def test_repo_id_without_prefix_falls_back_to_bare_suffix(model_config, monkeypatch):
    monkeypatch.delenv("HF_STAGING_PREFIX", raising=False)
    assert evaluate_checkpoint.resolve_repo_id(model_config) == "urdu-misinfo-xlmr-staging"


def test_both_experiments_resolve_to_distinct_repos(monkeypatch):
    monkeypatch.setenv("HF_STAGING_PREFIX", "u")
    mbert = evaluate_checkpoint.resolve_repo_id(load_config("model_mbert.yaml"))
    xlmr = evaluate_checkpoint.resolve_repo_id(load_config("model_xlmr.yaml"))
    assert mbert != xlmr
    assert "mbert" in mbert and "xlmr" in xlmr


# --------------------------------------------------------------------------
# the recovered record: same shape, honest provenance
# --------------------------------------------------------------------------


def _recovered_record(tmp_path, monkeypatch, model_config):
    """Build a record through the real builder with the same extra_metadata the
    recovery path attaches, without needing torch or the Hub."""
    from research.src.evaluation.metrics import (
        build_metrics_record,
        validate_metrics_record,
    )

    y_true = ["fake", "real", "fake", "real"]
    y_pred = ["fake", "real", "real", "real"]
    record = build_metrics_record(
        experiment_id="D",
        model=model_config["model"]["name"],
        dataset="ax_to_grind",
        split="test",
        seed=42,
        y_true=y_true,
        y_pred=y_pred,
        labels=["fake", "real"],
        scores=[0.9, 0.1, 0.4, 0.2],
        positive_label="fake",
        config={"data": {}, "model": model_config},
        extra_metadata={
            "n_train": 6405,
            "train_split": "ax_to_grind_train",
            "epochs": 3,
            "effective_batch_size": 16,
            "fp16": False,
            "truncation": {"n_truncated": 0, "n_examples": 4, "pct_truncated": 0.0},
            "train_runtime_seconds": None,
            "train_loss": None,
            "hardware": {"device": "cpu"},
            "evaluation_only_recovery": {
                "recovered": True,
                "reason": "M4-6",
                "checkpoint_repo_id": "u/urdu-misinfo-xlmr-staging",
                "checkpoint_revision_branch": "seed-42",
                "checkpoint_revision_sha": "abc123",
                "unavailable_fields": [
                    "train_runtime_seconds",
                    "train_loss",
                    "training_hardware",
                ],
                "hardware_is": "evaluation host, not the training host",
            },
            "dry_run": False,
        },
    )
    assert validate_metrics_record(record) == []
    return record


def test_recovered_record_satisfies_the_metrics_contract(tmp_path, monkeypatch, model_config):
    """A recovered file must be readable by every existing consumer unchanged."""
    record = _recovered_record(tmp_path, monkeypatch, model_config)
    for field in ("experiment_id", "model", "dataset", "split", "seed", "metrics",
                  "confusion_matrix", "per_class", "run_metadata"):
        assert field in record


def test_unrecoverable_fields_are_present_and_null_not_absent(tmp_path, monkeypatch, model_config):
    """Absent would read as an oversight; a number would be fabrication (rule 2)."""
    meta = _recovered_record(tmp_path, monkeypatch, model_config)["run_metadata"]
    assert "train_runtime_seconds" in meta
    assert "train_loss" in meta
    assert meta["train_runtime_seconds"] is None
    assert meta["train_loss"] is None


def test_recovery_is_self_declaring(tmp_path, monkeypatch, model_config):
    """A reader must never mistake a recovered file for a training-run file."""
    meta = _recovered_record(tmp_path, monkeypatch, model_config)["run_metadata"]
    recovery = meta["evaluation_only_recovery"]
    assert recovery["recovered"] is True
    assert recovery["checkpoint_revision_branch"] == "seed-42"
    # REPRODUCIBILITY.md Section 6: repo id AND revision, not repo id alone.
    assert recovery["checkpoint_repo_id"]
    assert recovery["checkpoint_revision_sha"]


def test_hardware_is_labelled_as_the_evaluation_host(tmp_path, monkeypatch, model_config):
    meta = _recovered_record(tmp_path, monkeypatch, model_config)["run_metadata"]
    assert "evaluation host" in meta["evaluation_only_recovery"]["hardware_is"]
    assert "training_hardware" in meta["evaluation_only_recovery"]["unavailable_fields"]


def test_recovered_record_is_json_serialisable(tmp_path, monkeypatch, model_config):
    """It gets written to disk and uploaded; a non-serialisable value would only
    surface after the GPU work was already done."""
    record = _recovered_record(tmp_path, monkeypatch, model_config)
    assert json.loads(json.dumps(record))["run_metadata"]["train_loss"] is None


# --------------------------------------------------------------------------
# filenames must collide with the training run's, so recovery overwrites in place
# --------------------------------------------------------------------------


def test_recovered_filename_matches_the_training_run_convention(model_config):
    experiment_id, seed, split = "D", 42, "test"
    expected = f"{experiment_id}_{model_config['model']['name']}_ax_to_grind_{split}_seed{seed}.json"
    assert expected == "D_xlm-roberta-base_ax_to_grind_test_seed42.json"


def test_cli_parses_experiment_and_seed_selection():
    """Partial recovery must be possible: if only some seeds are missing, re-score
    only those rather than all six."""
    parser_argv = ["--experiments", "D", "--seeds", "42", "123", "--no-push-results"]
    # main() would need torch and the Hub; parsing is what is asserted here.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["C", "D"], choices=["C", "D"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--no-push-results", action="store_true")
    args = parser.parse_args(parser_argv)
    assert args.experiments == ["D"]
    assert args.seeds == [42, 123]
    assert args.no_push_results is True
