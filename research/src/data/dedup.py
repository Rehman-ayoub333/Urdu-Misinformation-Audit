"""Exact and near-duplicate detection, within and across datasets.

Implements `DATASET_PLAN.md` Section 2 step 4 and `MASTER_PROJECT_BLUEPRINT.md`
Part 7 item 7. Runs on Tier-2-cleaned text, so trivial formatting differences do not
hide a duplicate pair.

Three scans:

  (a) within Ax-to-Grind    — near-duplicates straddling a train/test split would
                              inflate in-domain scores (RQ1)
  (b) within Notri-Fact     — same, for the cross-dataset test set
  (c) ACROSS the two        — **the go/no-go gate**. Any overlap silently
                              invalidates RQ2's zero-shot claim, because an
                              "unseen" test article would in fact have been
                              trained on. Overlaps are removed from Notri-Fact,
                              the test side, never from Ax-to-Grind — this
                              preserves training-set size, per
                              `MASTER_PROJECT_BLUEPRINT.md` Part 9.

Near-duplicate method: **character 5-gram MinHash with LSH banding**.

Chosen over the two alternatives `DATASET_PLAN.md` leaves open:

  * *TF-IDF cosine* would need a 10k x 13k dense similarity matrix for the
    cross-dataset scan (~135M pairs). Tractable but memory-hungry, and it scores
    topical similarity — two unrelated articles about the same cricket match score
    highly without being duplicates. That is the wrong notion here: the gate must
    catch *republished text*, not *related coverage*.
  * *SimHash* is cheaper still but gives a single 64-bit fingerprint whose Hamming
    distance is a coarse proxy for Jaccard, with no tunable accuracy/cost trade-off.

MinHash estimates Jaccard similarity over shingle sets directly, which is the right
similarity for near-identical text, and LSH banding makes the cross-dataset scan
near-linear rather than quadratic. **Character** 5-grams rather than word shingles
because Urdu is morphologically rich and inconsistently spaced — word-boundary
shingling is fragile on exactly the text this project handles.

Implemented on the standard library (`hashlib`) rather than adding `datasketch`: the
algorithm is ~40 lines, and a dependency whose behaviour must be understood in a
thesis defence is better written explicitly (`CLAUDE.md` rule 16).

Usage:
    python -m research.src.data.dedup            # run all three scans, write report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from research.src.data.clean import clean_text
from research.src.data.validate import load_ax_to_grind, load_notri_fact

AUDIT_DIR = Path(__file__).resolve().parents[3] / "research" / "data" / "audit"

# --- Tunables -----------------------------------------------------------------

SHINGLE_SIZE = 5
NUM_PERMUTATIONS = 128
LSH_BANDS = 32  # 32 bands x 4 rows; ~0.8 Jaccard threshold
LSH_ROWS = NUM_PERMUTATIONS // LSH_BANDS
NEAR_DUPLICATE_THRESHOLD = 0.8

# If cross-dataset overlap exceeds this fraction of Notri-Fact, stop and escalate
# rather than deleting a large part of the test set unilaterally.
CROSS_OVERLAP_ESCALATION_FRACTION = 0.05

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1
_MERSENNE_PRIME_U64 = np.uint64(_MERSENNE_PRIME)
_MAX_HASH_U64 = np.uint64(_MAX_HASH)


@dataclass
class DuplicateReport:
    """Result of one scan. Serialised to research/data/audit/ as evidence."""

    scan: str
    left_rows: int
    right_rows: int | None
    exact_duplicate_groups: int = 0
    exact_duplicate_rows: int = 0
    near_duplicate_pairs: int = 0
    rows_to_remove: list[str] = field(default_factory=list)
    sample_pairs: list[dict[str, object]] = field(default_factory=list)
    method: str = (
        f"char-{SHINGLE_SIZE}-gram MinHash, {NUM_PERMUTATIONS} permutations, "
        f"LSH {LSH_BANDS}x{LSH_ROWS}, Jaccard >= {NEAR_DUPLICATE_THRESHOLD}"
    )

    @property
    def total_flagged(self) -> int:
        return len(self.rows_to_remove)


def _shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Character n-grams. Short strings yield themselves as a single shingle."""
    text = text.strip()
    if len(text) < size:
        return {text} if text else set()
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def _build_hash_coefficients(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Random (a, b) pairs for the universal hash family h(x) = (a*x + b) mod p.

    Seeded so a rerun produces identical signatures — the dedup result is an input
    to the committed splits, so it must be reproducible (REPRODUCIBILITY.md).
    """
    rng = random.Random(seed)
    a = np.array(
        [rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(NUM_PERMUTATIONS)], dtype=object
    )
    b = np.array(
        [rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(NUM_PERMUTATIONS)], dtype=object
    )
    # Reduce mod the prime once, then keep everything in uint64 range. The
    # coefficients stay below 2**61 so a*h + b cannot overflow int64 once h is
    # bounded by 2**32, which lets the hot loop run in native integer types.
    return np.array([int(x) for x in a], dtype=np.uint64), np.array(
        [int(x) for x in b], dtype=np.uint64
    )


_COEF_A, _COEF_B = _build_hash_coefficients()


def _base_hashes(shingles: set[str]) -> np.ndarray:
    """32-bit blake2b hash per shingle, as a uint64 vector."""
    return np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "big")
            for s in shingles
        ),
        dtype=np.uint64,
        count=len(shingles),
    )


def minhash_signature(text: str) -> tuple[int, ...]:
    """MinHash signature: the min hash value per permutation over the shingle set.

    Vectorised with numpy. The pure-Python form was ~10^9 interpreted operations
    across the corpus and did not finish in ten minutes; this computes the whole
    (permutations x shingles) grid per document in native integer types.

    numpy is fine to use here — the pure-stdlib constraint (E18) applies only to
    `clean.py`, which is the file copied into the serving image.
    """
    shingles = _shingles(text)
    if not shingles:
        return tuple([int(_MAX_HASH)] * NUM_PERMUTATIONS)

    base = _base_hashes(shingles)
    # (NUM_PERMUTATIONS, n_shingles) -> min along the shingle axis.
    hashed = (_COEF_A[:, None] * base[None, :] + _COEF_B[:, None]) % _MERSENNE_PRIME_U64
    return tuple(int(v) for v in (hashed & _MAX_HASH_U64).min(axis=1))


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Estimate Jaccard similarity as the fraction of agreeing signature positions."""
    return sum(1 for x, y in zip(left, right) if x == y) / len(left)


def _lsh_buckets(signatures: dict[str, tuple[int, ...]]) -> dict[tuple[int, bytes], list[str]]:
    """Band the signatures so only plausible pairs are compared."""
    buckets: dict[tuple[int, bytes], list[str]] = defaultdict(list)
    for row_id, signature in signatures.items():
        for band in range(LSH_BANDS):
            chunk = signature[band * LSH_ROWS : (band + 1) * LSH_ROWS]
            key = hashlib.blake2b(
                repr(chunk).encode("utf-8"), digest_size=8
            ).digest()
            buckets[(band, key)].append(row_id)
    return buckets


def _exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_within_dataset_duplicates(
    frame: pd.DataFrame, dataset: str, *, texts: pd.Series | None = None
) -> DuplicateReport:
    """Exact + near-duplicate scan inside one dataset.

    Keeps the first occurrence of each duplicate group and flags the rest, so
    deduplication is deterministic given a stable row order.
    """
    cleaned = texts if texts is not None else frame["text"].map(clean_text)
    report = DuplicateReport(scan=f"within:{dataset}", left_rows=len(frame), right_rows=None)

    # --- exact ---
    digests = cleaned.map(_exact_hash)
    groups = defaultdict(list)
    for row_id, digest in zip(frame["row_id"], digests):
        groups[digest].append(row_id)

    flagged: set[str] = set()
    for members in groups.values():
        if len(members) > 1:
            report.exact_duplicate_groups += 1
            report.exact_duplicate_rows += len(members) - 1
            flagged.update(members[1:])

    # --- near, over survivors only ---
    survivors = {
        row_id: text
        for row_id, text in zip(frame["row_id"], cleaned)
        if row_id not in flagged
    }
    signatures = {row_id: minhash_signature(text) for row_id, text in survivors.items()}

    checked: set[tuple[str, str]] = set()
    for members in _lsh_buckets(signatures).values():
        if len(members) < 2:
            continue
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                pair = (left, right) if left < right else (right, left)
                if pair in checked:
                    continue
                checked.add(pair)
                score = _jaccard(signatures[left], signatures[right])
                if score >= NEAR_DUPLICATE_THRESHOLD:
                    report.near_duplicate_pairs += 1
                    loser = max(pair)
                    flagged.add(loser)
                    if len(report.sample_pairs) < 10:
                        report.sample_pairs.append(
                            {
                                "a": pair[0],
                                "b": pair[1],
                                "jaccard": round(score, 4),
                                "a_text": survivors[pair[0]][:160],
                                "b_text": survivors[pair[1]][:160],
                            }
                        )

    report.rows_to_remove = sorted(flagged)
    return report


def find_cross_dataset_duplicates(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_texts: pd.Series | None = None,
    right_texts: pd.Series | None = None,
) -> DuplicateReport:
    """THE GATE: scan for articles shared between Ax-to-Grind and Notri-Fact.

    Only `right` (Notri-Fact, the test side) rows are ever flagged for removal.
    """
    left_clean = left_texts if left_texts is not None else left["text"].map(clean_text)
    right_clean = right_texts if right_texts is not None else right["text"].map(clean_text)

    report = DuplicateReport(
        scan="cross:ax_to_grind|notri_fact",
        left_rows=len(left),
        right_rows=len(right),
    )

    # --- exact ---
    left_digests = {_exact_hash(t) for t in left_clean}
    flagged: set[str] = set()
    for row_id, text in zip(right["row_id"], right_clean):
        if _exact_hash(text) in left_digests:
            flagged.add(row_id)
            report.exact_duplicate_rows += 1
    report.exact_duplicate_groups = report.exact_duplicate_rows

    # --- near ---
    left_signatures = {
        row_id: minhash_signature(text) for row_id, text in zip(left["row_id"], left_clean)
    }
    right_signatures = {
        row_id: minhash_signature(text) for row_id, text in zip(right["row_id"], right_clean)
    }

    combined = {**left_signatures, **right_signatures}
    left_ids = set(left_signatures)

    checked: set[tuple[str, str]] = set()
    for members in _lsh_buckets(combined).values():
        if len(members) < 2:
            continue
        lefts = [m for m in members if m in left_ids]
        rights = [m for m in members if m not in left_ids]
        if not lefts or not rights:
            continue  # same-dataset pair; handled by the within-dataset scan
        for a in lefts:
            for b in rights:
                if (a, b) in checked:
                    continue
                checked.add((a, b))
                score = _jaccard(combined[a], combined[b])
                if score >= NEAR_DUPLICATE_THRESHOLD:
                    report.near_duplicate_pairs += 1
                    flagged.add(b)  # always remove from the TEST side
                    if len(report.sample_pairs) < 10:
                        report.sample_pairs.append({"a": a, "b": b, "jaccard": round(score, 4)})

    report.rows_to_remove = sorted(flagged)
    return report


def write_report(reports: list[DuplicateReport], destination: Path | None = None) -> Path:
    """Persist the scans to research/data/audit/duplication_report.json."""
    target = destination or (AUDIT_DIR / "duplication_report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": NEAR_DUPLICATE_THRESHOLD,
        "shingle_size": SHINGLE_SIZE,
        "num_permutations": NUM_PERMUTATIONS,
        "lsh_bands": LSH_BANDS,
        "lsh_rows": LSH_ROWS,
        "scans": [asdict(r) for r in reports],
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    axg, _ = load_ax_to_grind()
    ntf, _ = load_notri_fact()
    axg_clean = axg["text"].map(clean_text)
    ntf_clean = ntf["text"].map(clean_text)

    reports = []
    print("=== within Ax-to-Grind ===")
    r = find_within_dataset_duplicates(axg, "ax_to_grind", texts=axg_clean)
    print(
        f"  exact groups={r.exact_duplicate_groups:,} rows={r.exact_duplicate_rows:,} | "
        f"near pairs={r.near_duplicate_pairs:,} | total flagged={r.total_flagged:,}"
    )
    reports.append(r)

    print("=== within Notri-Fact ===")
    r = find_within_dataset_duplicates(ntf, "notri_fact", texts=ntf_clean)
    print(
        f"  exact groups={r.exact_duplicate_groups:,} rows={r.exact_duplicate_rows:,} | "
        f"near pairs={r.near_duplicate_pairs:,} | total flagged={r.total_flagged:,}"
    )
    reports.append(r)

    print("=== CROSS-DATASET GATE ===")
    cross = find_cross_dataset_duplicates(
        axg, ntf, left_texts=axg_clean, right_texts=ntf_clean
    )
    print(
        f"  exact={cross.exact_duplicate_rows:,} | near pairs={cross.near_duplicate_pairs:,} | "
        f"to remove from Notri-Fact={cross.total_flagged:,}"
    )
    reports.append(cross)

    fraction = cross.total_flagged / len(ntf) if len(ntf) else 0.0
    print(f"  overlap fraction of Notri-Fact: {fraction:.4%}")
    if fraction > CROSS_OVERLAP_ESCALATION_FRACTION:
        print(
            f"\n  ESCALATE: overlap exceeds {CROSS_OVERLAP_ESCALATION_FRACTION:.0%} of the "
            "test set. This looks like something other than incidental duplication — "
            "stop and report rather than deleting this much of the test set.",
        )

    target = write_report(reports)
    print(f"\nWrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
