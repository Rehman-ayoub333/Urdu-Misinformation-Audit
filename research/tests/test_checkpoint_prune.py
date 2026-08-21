"""Tests for verified-then-prune of local checkpoints.

Fifteen XLM-R checkpoints at ~1.1 GB is ~16.5 GB against Kaggle's ~19.5 GB
working quota, so Experiment I cannot finish without freeing each run's local copy
as it goes. The danger is that a disk fix is one mistake away from recreating
`DECISION_REGISTER.md` **M4-6** — a completed training run whose only artefact is
gone — while looking like harmless housekeeping.

So the property under test is narrow and absolute: **the local copy is deleted
only after the Hub has been asked, and has answered that a complete checkpoint is
there.** Attempted, skipped, dry-run, partial and unverifiable pushes must all
leave it alone. Every test here that asserts deletion has a mirror-image test that
asserts non-deletion, because only the pair is meaningful.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from research.src.models.checkpoint_artifacts import (
    CONFIG_FILES,
    TOKENIZER_FILES,
    WEIGHT_FILES,
    missing_artifact_kinds,
)
from research.src.models.transformer import (
    prune_local_checkpoint,
    push_verify_prune,
    verify_checkpoint_uploaded,
)

COMPLETE = {"config.json", "model.safetensors", "tokenizer.json", "special_tokens_map.json"}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A stand-in for a finished run's local checkpoint directory."""
    directory = tmp_path / "D_xlm-roberta-base_seed42"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"x" * 2048)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def _hub_stub(monkeypatch: pytest.MonkeyPatch, *, files=None, raises=None, sha="abc123"):
    """Patch huggingface_hub.HfApi so no network call is made."""
    import huggingface_hub

    class _Api:
        def __init__(self, token=None):
            self.token = token

        def repo_info(self, repo_id, revision=None, **kwargs):
            if raises is not None:
                raise raises
            siblings = [types.SimpleNamespace(rfilename=name) for name in (files or [])]
            return types.SimpleNamespace(siblings=siblings, sha=sha)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)


# --------------------------------------------------------------------------
# the completeness definition is shared, not restated
# --------------------------------------------------------------------------


def test_complete_listing_has_no_missing_kinds() -> None:
    assert missing_artifact_kinds(COMPLETE) == []


@pytest.mark.parametrize(
    ("drop", "expected"),
    [
        ({"config.json"}, ["config"]),
        ({"model.safetensors"}, ["weights"]),
        ({"tokenizer.json"}, ["tokenizer"]),
    ],
)
def test_each_missing_group_is_named(drop: set[str], expected: list[str]) -> None:
    """A branch with weights but no tokenizer is not re-scorable (M4-6)."""
    assert missing_artifact_kinds(COMPLETE - drop) == expected


def test_the_inventory_script_uses_the_same_definition() -> None:
    """If the auditor and the prune disagreed, the prune could delete a checkpoint
    the auditor would afterwards call incomplete."""
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_inventory", root / "research" / "scripts" / "inventory_staging_checkpoints.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Identity, not equality: the auditor must be using these very objects, so a
    # future edit to one definition cannot leave the other behind.
    assert module.WEIGHT_FILES is WEIGHT_FILES
    assert module.TOKENIZER_FILES is TOKENIZER_FILES
    assert module.missing_artifact_kinds is missing_artifact_kinds
    # CONFIG_FILES is reached through missing_artifact_kinds rather than directly.
    assert CONFIG_FILES == ("config.json",)


# --------------------------------------------------------------------------
# verification fails closed
# --------------------------------------------------------------------------


def test_complete_checkpoint_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_stub(monkeypatch, files=COMPLETE)
    result = verify_checkpoint_uploaded("user/repo", "seed-42", token="t")
    assert result["verified"] is True
    assert result["revision_sha"] == "abc123"


@pytest.mark.parametrize("drop", ["config.json", "model.safetensors", "tokenizer.json"])
def test_incomplete_checkpoint_does_not_verify(
    monkeypatch: pytest.MonkeyPatch, drop: str
) -> None:
    _hub_stub(monkeypatch, files=COMPLETE - {drop})
    result = verify_checkpoint_uploaded("user/repo", "seed-42", token="t")
    assert result["verified"] is False
    assert result["missing"]


def test_unreachable_hub_does_not_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inability to check must never read as permission to delete."""
    _hub_stub(monkeypatch, raises=ConnectionError("hub is down"))
    result = verify_checkpoint_uploaded("user/repo", "seed-42", token="t")
    assert result["verified"] is False
    assert "hub is down" in result["reason"]


def test_empty_branch_does_not_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_stub(monkeypatch, files=[])
    assert verify_checkpoint_uploaded("user/repo", "seed-42", token="t")["verified"] is False


# --------------------------------------------------------------------------
# prune fires ONLY on a verified push — the pair that matters
# --------------------------------------------------------------------------


def test_verified_push_prunes_the_local_copy(run_dir: Path) -> None:
    result = prune_local_checkpoint(run_dir, {"verified": True, "repo_id": "r", "revision": "b"})
    assert result["pruned"] is True
    assert not run_dir.exists()
    assert result["freed_bytes"] > 0


def test_unverified_push_leaves_the_local_copy(run_dir: Path) -> None:
    """THE case this whole mechanism exists for: the upload did not land, so the
    local directory may be the only copy of a finished training run."""
    result = prune_local_checkpoint(run_dir, {"verified": False, "reason": "missing ['weights']"})
    assert result["pruned"] is False
    assert run_dir.exists(), "an unverified push must never delete the local checkpoint"
    assert "NOT verified" in result["reason"]


def test_absent_verification_leaves_the_local_copy(run_dir: Path) -> None:
    """No verification performed is not the same as verification passed."""
    assert prune_local_checkpoint(run_dir, None)["pruned"] is False
    assert run_dir.exists()


def test_malformed_verification_leaves_the_local_copy(run_dir: Path) -> None:
    """A dict without the key must not be read as truthy-by-accident."""
    assert prune_local_checkpoint(run_dir, {"reason": "who knows"})["pruned"] is False
    assert run_dir.exists()


def test_no_run_dir_is_a_no_op(tmp_path: Path) -> None:
    assert prune_local_checkpoint(None, {"verified": True})["pruned"] is False


def test_already_absent_directory_is_reported_not_raised(tmp_path: Path) -> None:
    result = prune_local_checkpoint(tmp_path / "gone", {"verified": True})
    assert result["pruned"] is False
    assert result["reason"] == "already absent"


# --------------------------------------------------------------------------
# the full push -> verify -> prune path
# --------------------------------------------------------------------------


def _patch_push(monkeypatch: pytest.MonkeyPatch, push_result: dict) -> None:
    from research.src.models import transformer

    monkeypatch.setattr(
        transformer, "push_checkpoint", lambda *a, **k: dict(push_result)
    )


def test_end_to_end_verified_push_prunes(
    monkeypatch: pytest.MonkeyPatch, run_dir: Path
) -> None:
    _patch_push(
        monkeypatch,
        {"pushed": True, "repo_id": "user/repo", "revision_branch": "seed-42-punct-ablation"},
    )
    _hub_stub(monkeypatch, files=COMPLETE)

    info = push_verify_prune(None, None, {}, 42, run_dir=run_dir)

    assert info["verification"]["verified"] is True
    assert info["prune"]["pruned"] is True
    assert not run_dir.exists()


def test_end_to_end_unverified_push_keeps(
    monkeypatch: pytest.MonkeyPatch, run_dir: Path
) -> None:
    """Push reported success, but the Hub does not hold a complete checkpoint."""
    _patch_push(
        monkeypatch, {"pushed": True, "repo_id": "user/repo", "revision_branch": "seed-42"}
    )
    _hub_stub(monkeypatch, files=COMPLETE - {"model.safetensors"})

    info = push_verify_prune(None, None, {}, 42, run_dir=run_dir)

    assert info["verification"]["verified"] is False
    assert info["prune"]["pruned"] is False
    assert run_dir.exists(), "a push the Hub cannot confirm must not delete anything"


def test_end_to_end_unreachable_hub_keeps(
    monkeypatch: pytest.MonkeyPatch, run_dir: Path
) -> None:
    _patch_push(
        monkeypatch, {"pushed": True, "repo_id": "user/repo", "revision_branch": "seed-42"}
    )
    _hub_stub(monkeypatch, raises=TimeoutError("no route to host"))

    info = push_verify_prune(None, None, {}, 42, run_dir=run_dir)

    assert info["prune"]["pruned"] is False
    assert run_dir.exists()


def test_dry_run_never_prunes(monkeypatch: pytest.MonkeyPatch, run_dir: Path) -> None:
    """A dry run pushes nothing, so the local copy is the only one that exists."""
    _patch_push(monkeypatch, {"pushed": False, "reason": "dry run"})

    info = push_verify_prune(None, None, {}, 42, run_dir=run_dir, dry_run=True)

    assert info["verification"]["verified"] is False
    assert info["prune"]["pruned"] is False
    assert run_dir.exists()


def test_push_disabled_never_prunes(monkeypatch: pytest.MonkeyPatch, run_dir: Path) -> None:
    _patch_push(monkeypatch, {"pushed": False, "reason": "pushing disabled"})
    info = push_verify_prune(None, None, {}, 42, run_dir=run_dir, dry_run=True)
    assert info["prune"]["pruned"] is False
    assert run_dir.exists()


def test_prune_can_be_turned_off_even_when_verified(
    monkeypatch: pytest.MonkeyPatch, run_dir: Path
) -> None:
    _patch_push(
        monkeypatch, {"pushed": True, "repo_id": "user/repo", "revision_branch": "seed-42"}
    )
    _hub_stub(monkeypatch, files=COMPLETE)

    info = push_verify_prune(None, None, {}, 42, run_dir=run_dir, prune=False)

    assert info["verification"]["verified"] is True
    assert info["prune"]["pruned"] is False
    assert run_dir.exists()


# --------------------------------------------------------------------------
# every training experiment takes the same path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "research/src/models/transformer.py",            # C, D in-domain
        "research/src/experiments/punctuation_ablation.py",  # Q
        "research/src/experiments/length_ablation.py",       # I
    ],
)
def test_every_training_runner_uses_the_shared_path(module_path: str) -> None:
    """The delete-only-if-verified rule must not hold in one experiment and not
    another. Asserted structurally: no training runner may call `push_checkpoint`
    directly and skip the verification step."""
    root = Path(__file__).resolve().parents[2]
    source = (root / module_path).read_text(encoding="utf-8")

    assert "push_verify_prune(" in source, f"{module_path} does not use the shared path"

    # `transformer.py` legitimately defines and calls push_checkpoint inside
    # push_verify_prune; the experiment runners must not call it at all.
    if "experiments" in module_path:
        assert "push_checkpoint(" not in source, (
            f"{module_path} calls push_checkpoint directly, bypassing verification"
        )


# --------------------------------------------------------------------------
# the RE-SCORE paths have nothing to prune, and that is a property worth pinning
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "research/src/models/evaluate_checkpoint.py",     # notebook 06 section a
        "research/src/experiments/run_cross_dataset.py",  # notebook 06 section b
    ],
)
def test_rescore_paths_push_no_checkpoint_and_keep_no_local_copy(module_path: str) -> None:
    """Sections a and b need no prune because they create nothing to prune.

    This is the honest reason those paths differ from the training ones, and it is
    asserted rather than assumed. They *download* checkpoints (into the shared HF
    cache, which is not this project's to delete — section b reuses what section a
    fetched) and they run the Trainer inside a `tempfile.TemporaryDirectory`, so
    the scoring scratch is released by the interpreter regardless of how the run
    ends. Neither pushes a checkpoint, so there is never a Hub copy to verify
    against, and adding a prune call there would be a no-op dressed as coverage.

    If either ever starts pushing checkpoints, this test fails and the prune must
    be wired in — which is the point of pinning it.
    """
    root = Path(__file__).resolve().parents[2]
    source = (root / module_path).read_text(encoding="utf-8")

    assert "push_checkpoint(" not in source, f"{module_path} now pushes checkpoints"
    assert "push_verify_prune(" not in source, f"{module_path} now pushes checkpoints"
    assert "CHECKPOINT_DIR" not in source, (
        f"{module_path} writes to the persistent checkpoint directory"
    )


def test_scoring_scratch_is_a_temporary_directory() -> None:
    """The mechanism behind the test above, named explicitly.

    `score_frame` is the single scoring implementation both re-score paths use.
    Its Trainer needs an `output_dir`; that it is a TemporaryDirectory rather than
    `checkpoints/` is what makes sections a and b disk-neutral.
    """
    root = Path(__file__).resolve().parents[2]
    source = (root / "research/src/models/evaluate_checkpoint.py").read_text(encoding="utf-8")

    assert "TemporaryDirectory()" in source
    assert "output_dir=tmp_dir" in source
