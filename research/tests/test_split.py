"""Tests for the committed train/val/test splits.

The split index files are committed and every later milestone trains against them,
so a defect here silently corrupts every downstream result. These assert the
properties that make the splits trustworthy: disjointness, determinism,
stratification, and that deduplication really was applied before splitting.
"""

from __future__ import annotations

import json

import pytest

from research.src.data.split import (
    SPLIT_SEED,
    SPLITS_DIR,
    stratified_split,
    read_split,
)
from research.src.data.validate import (
    AX_TO_GRIND_SOURCE,
    NOTRI_FACT_SOURCE,
    load_ax_to_grind,
)

SPLIT_NAMES = ("ax_to_grind_train", "ax_to_grind_val", "ax_to_grind_test")


def _require_splits() -> None:
    if not (SPLITS_DIR / "splits_manifest.json").exists():
        pytest.skip("splits not generated — run research/src/data/split.py")


@pytest.fixture(scope="module")
def manifest() -> dict:
    _require_splits()
    return json.loads((SPLITS_DIR / "splits_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def splits() -> dict[str, list[str]]:
    _require_splits()
    return {name: read_split(name) for name in SPLIT_NAMES}


# --- Structural invariants -----------------------------------------------------


def test_splits_are_disjoint(splits: dict[str, list[str]]) -> None:
    """No row may appear in more than one split — the basic leakage check."""
    train, val, test = (set(splits[n]) for n in SPLIT_NAMES)
    assert not train & val, f"train/val overlap: {sorted(train & val)[:5]}"
    assert not train & test, f"train/test overlap: {sorted(train & test)[:5]}"
    assert not val & test, f"val/test overlap: {sorted(val & test)[:5]}"


def test_no_duplicate_row_ids_within_a_split(splits: dict[str, list[str]]) -> None:
    for name, ids in splits.items():
        assert len(ids) == len(set(ids)), f"{name} contains repeated row_ids"


def test_split_sizes_match_manifest(splits: dict[str, list[str]], manifest: dict) -> None:
    for name in SPLIT_NAMES:
        assert len(splits[name]) == manifest["counts"][name]


def test_split_proportions_are_70_15_15(splits: dict[str, list[str]]) -> None:
    total = sum(len(splits[n]) for n in SPLIT_NAMES)
    assert len(splits["ax_to_grind_train"]) / total == pytest.approx(0.70, abs=0.01)
    assert len(splits["ax_to_grind_val"]) / total == pytest.approx(0.15, abs=0.01)
    assert len(splits["ax_to_grind_test"]) / total == pytest.approx(0.15, abs=0.01)


def test_notri_fact_holdout_is_never_split() -> None:
    """R3/R7: the cross-dataset test set is used whole, never partitioned."""
    _require_splits()
    holdout = read_split("notri_fact_holdout")
    assert holdout, "holdout index is empty"
    assert not (SPLITS_DIR / "notri_fact_train.txt").exists()
    assert not (SPLITS_DIR / "notri_fact_val.txt").exists()
    assert not (SPLITS_DIR / "notri_fact_test.txt").exists()


def test_no_notri_fact_row_appears_in_an_ax_to_grind_split(
    splits: dict[str, list[str]],
) -> None:
    """R7: the two pools are never merged. Guards the id-prefix boundary."""
    for name, ids in splits.items():
        stray = [i for i in ids if not i.startswith("axg-")]
        assert not stray, f"{name} contains non-Ax-to-Grind ids: {stray[:5]}"


def test_holdout_contains_only_notri_fact_rows() -> None:
    _require_splits()
    stray = [i for i in read_split("notri_fact_holdout") if not i.startswith("ntf-")]
    assert not stray, f"holdout contains non-Notri-Fact ids: {stray[:5]}"


# --- Deduplication was applied BEFORE splitting --------------------------------


def test_deduplication_was_applied_before_splitting(manifest: dict) -> None:
    """Part 8 requires dedup before splitting so no near-duplicate straddles splits."""
    assert manifest["deduplication"]["applied_before_split"] is True
    assert manifest["deduplication"]["ax_to_grind_rows_removed"] > 0


def test_removed_duplicates_are_absent_from_every_split(
    splits: dict[str, list[str]], manifest: dict
) -> None:
    """The split union must be strictly smaller than the raw dataset.

    If dedup had been skipped or applied after splitting, the union would equal the
    full row count instead.
    """
    if not AX_TO_GRIND_SOURCE.exists():
        pytest.skip("raw data not present")
    frame, _ = load_ax_to_grind()
    union = set().union(*(set(splits[n]) for n in SPLIT_NAMES))

    assert len(union) == manifest["counts"]["ax_to_grind_total_after_dedup"]
    assert len(union) < len(frame), (
        "the split union equals the un-deduplicated row count — dedup was not applied"
    )
    removed = manifest["deduplication"]["ax_to_grind_rows_removed"]
    assert len(frame) - len(union) == removed


def test_cross_dataset_gate_recorded_as_passing(manifest: dict) -> None:
    """Splits must never be written while cross-dataset overlap exists."""
    assert manifest["deduplication"]["cross_dataset_overlap"] == 0


# --- Stratification and determinism --------------------------------------------


def test_stratification_preserves_class_balance(manifest: dict) -> None:
    """Each split's class balance must track the corpus balance (~50/50)."""
    for name in SPLIT_NAMES:
        counts = manifest["label_counts"][name]
        total = sum(counts.values())
        for label, count in counts.items():
            assert count / total == pytest.approx(0.5, abs=0.02), (
                f"{name} is imbalanced for {label}: {counts}"
            )


def test_split_is_deterministic_under_the_recorded_seed() -> None:
    """Re-running the split must reproduce the committed index files exactly.

    This is what makes the committed files meaningful: another researcher running
    split.py gets the same partition rather than a different one that happens to
    have the same proportions.
    """
    if not (AX_TO_GRIND_SOURCE.exists() and NOTRI_FACT_SOURCE.exists()):
        pytest.skip("raw data not present")
    _require_splits()

    frame, _ = load_ax_to_grind()
    committed_union = set().union(*(set(read_split(n)) for n in SPLIT_NAMES))
    deduped = frame.loc[frame["row_id"].isin(committed_union)].reset_index(drop=True)

    first = stratified_split(deduped, seed=SPLIT_SEED)
    second = stratified_split(deduped, seed=SPLIT_SEED)
    for name in ("train", "val", "test"):
        assert first[name].row_ids == second[name].row_ids, f"{name} is not deterministic"
        assert first[name].row_ids == read_split(f"ax_to_grind_{name}"), (
            f"regenerating {name} does not reproduce the committed index file"
        )


def test_a_different_seed_produces_a_different_partition() -> None:
    """Guards against a split that ignores its seed and is accidentally fixed."""
    if not AX_TO_GRIND_SOURCE.exists():
        pytest.skip("raw data not present")
    frame, _ = load_ax_to_grind()
    subset = frame.head(2000)
    assert (
        stratified_split(subset, seed=SPLIT_SEED)["train"].row_ids
        != stratified_split(subset, seed=SPLIT_SEED + 1)["train"].row_ids
    )
