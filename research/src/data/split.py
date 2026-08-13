"""Build the committed train/val/test splits.

Implements `DATASET_PLAN.md` Section 2 step 6 and `MASTER_PROJECT_BLUEPRINT.md`
Part 9.

  * **Ax-to-Grind** — stratified 70/15/15 train/val/test by label, applied
    **after** deduplication so no near-duplicate pair straddles train and test.
  * **Notri-Fact** — held out **in full**, never split, never trained on. Written
    as a single index file so downstream code loads it the same way as any other
    split rather than special-casing it (`DECISION_REGISTER.md` R3/R7).

**Grouped splitting is not used, on evidence rather than by default.** Part 9 says
to group by publisher *if a source field survives the audit*. It does not:
Ax-to-Grind ships only `Sr. No.`, `News Items`, `Label` (`DECISION_REGISTER.md`
M2-2), so there is nothing to group on. Plain stratification is the honest
fallback, and the residual source-leakage risk stays untestable — recorded as a
limitation, not silently treated as absent.

**Deduplication is applied before splitting, but only within Ax-to-Grind.** The
cross-dataset scan found zero overlap (`research/data/audit/duplication_report.json`),
so no Notri-Fact row is dropped; had any been found, they would have been removed
from Notri-Fact, the test side, never from the training side (Part 9).

Output: `research/data/splits/{ax_to_grind_{train,val,test},notri_fact_holdout}.txt`,
one `row_id` per line, plus `splits_manifest.json`. Row-ID index files rather than
copies of the data — small enough to commit, and they make the split reproducible
for another researcher without redistributing the corpus (`GITHUB_PLAN.md` S2).

Usage:
    python -m research.src.data.split
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.src.data.clean import clean_text
from research.src.data.dedup import (
    find_cross_dataset_duplicates,
    find_within_dataset_duplicates,
)
from research.src.data.validate import load_ax_to_grind, load_notri_fact

_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = _ROOT / "research" / "data" / "splits"

# REPRODUCIBILITY.md Section 2: seed recorded explicitly, never a library default.
SPLIT_SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Grouping field, if one is ever added upstream. None today — see M2-2.
GROUP_FIELD: str | None = None


@dataclass
class SplitResult:
    name: str
    row_ids: list[str]
    label_counts: dict[str, int]


def stratified_split(
    frame: pd.DataFrame, *, seed: int = SPLIT_SEED
) -> dict[str, SplitResult]:
    """Label-stratified 70/15/15 split.

    Stratification is applied per label independently, so each split preserves the
    corpus class balance. Assignment is by shuffled position within each label
    group under a fixed seed, which makes the result deterministic given the same
    input rows — the property that lets the index files be committed and trusted.
    """
    rng = np.random.default_rng(seed)
    assignments: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for label in sorted(frame["label"].unique()):
        ids = frame.loc[frame["label"] == label, "row_id"].to_numpy()
        ids = np.sort(ids)  # stable starting order regardless of upstream row order
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(round(n * TRAIN_FRACTION))
        n_val = int(round(n * VAL_FRACTION))
        assignments["train"].extend(ids[:n_train].tolist())
        assignments["val"].extend(ids[n_train : n_train + n_val].tolist())
        assignments["test"].extend(ids[n_train + n_val :].tolist())

    label_by_id = dict(zip(frame["row_id"], frame["label"]))
    return {
        name: SplitResult(
            name=name,
            row_ids=sorted(ids),
            label_counts=dict(pd.Series([label_by_id[i] for i in ids]).value_counts()),
        )
        for name, ids in assignments.items()
    }


def write_split(name: str, row_ids: list[str], destination: Path | None = None) -> Path:
    target = (destination or SPLITS_DIR) / f"{name}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(row_ids) + "\n", encoding="utf-8")
    return target


def read_split(name: str, source: Path | None = None) -> list[str]:
    target = (source or SPLITS_DIR) / f"{name}.txt"
    return [line for line in target.read_text(encoding="utf-8").splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    axg, _ = load_ax_to_grind()
    ntf, _ = load_notri_fact()
    axg_clean = axg["text"].map(clean_text)
    ntf_clean = ntf["text"].map(clean_text)

    print("=== deduplication (before splitting, per Part 8) ===")
    axg_dupes = find_within_dataset_duplicates(axg, "ax_to_grind", texts=axg_clean)
    ntf_dupes = find_within_dataset_duplicates(ntf, "notri_fact", texts=ntf_clean)
    cross = find_cross_dataset_duplicates(
        axg, ntf, left_texts=axg_clean, right_texts=ntf_clean
    )

    if cross.total_flagged:
        print(
            f"  GATE FAILED: {cross.total_flagged} cross-dataset duplicates. "
            "Refusing to write splits.",
        )
        return 1
    print(f"  cross-dataset overlap: {cross.total_flagged} (gate passed)")

    axg_removed = set(axg_dupes.rows_to_remove)
    ntf_removed = set(ntf_dupes.rows_to_remove)
    print(f"  ax_to_grind: {len(axg_removed):,} duplicate rows removed")
    print(f"  notri_fact : {len(ntf_removed):,} duplicate rows removed")

    axg_dedup = axg.loc[~axg["row_id"].isin(axg_removed)].reset_index(drop=True)
    ntf_dedup = ntf.loc[~ntf["row_id"].isin(ntf_removed)].reset_index(drop=True)

    print("\n=== Ax-to-Grind stratified 70/15/15 ===")
    if GROUP_FIELD is None:
        print(
            "  grouping: NONE — Ax-to-Grind has no source/publisher field "
            "(DECISION_REGISTER.md M2-2), so plain stratification is used."
        )
    splits = stratified_split(axg_dedup)
    written = []
    for name in ("train", "val", "test"):
        result = splits[name]
        path = write_split(f"ax_to_grind_{name}", result.row_ids)
        written.append(path)
        share = len(result.row_ids) / len(axg_dedup)
        print(
            f"  {name:<6} n={len(result.row_ids):>6,} ({share:6.2%})  {result.label_counts}"
        )

    print("\n=== Notri-Fact holdout (never split, never trained on) ===")
    holdout_ids = sorted(ntf_dedup["row_id"].tolist())
    written.append(write_split("notri_fact_holdout", holdout_ids))
    print(f"  holdout n={len(holdout_ids):,}  {dict(ntf_dedup['label'].value_counts())}")

    manifest = {
        "seed": SPLIT_SEED,
        "fractions": {
            "train": TRAIN_FRACTION,
            "val": VAL_FRACTION,
            "test": TEST_FRACTION,
        },
        "grouping_field": GROUP_FIELD,
        "grouping_note": (
            "Plain stratification. Blueprint Part 9's grouped-splitting option "
            "requires a source/publisher field, which Ax-to-Grind does not have "
            "(DECISION_REGISTER.md M2-2). Residual source-leakage risk is therefore "
            "untestable for this dataset and must be stated as a limitation."
        ),
        "deduplication": {
            "applied_before_split": True,
            "ax_to_grind_rows_removed": len(axg_removed),
            "notri_fact_rows_removed": len(ntf_removed),
            "cross_dataset_overlap": cross.total_flagged,
        },
        "counts": {
            "ax_to_grind_total_after_dedup": len(axg_dedup),
            **{f"ax_to_grind_{n}": len(splits[n].row_ids) for n in ("train", "val", "test")},
            "notri_fact_holdout": len(holdout_ids),
        },
        "label_counts": {
            f"ax_to_grind_{n}": {k: int(v) for k, v in splits[n].label_counts.items()}
            for n in ("train", "val", "test")
        },
    }
    manifest_path = SPLITS_DIR / "splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    for path in [*written, manifest_path]:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
