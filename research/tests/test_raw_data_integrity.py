"""Milestone 1 data-validation tests: file presence and checksum match.

Scope is deliberately narrow. `ROADMAP.md` Milestone 1 requires "a data-validation
test confirming file presence + checksum match" — that and nothing more. Schema
checks, label-value checks, null-text checks and row-count-against-documented-size
checks belong to `validate.py` at Milestone 2 (`DATASET_PLAN.md` Section 2 step 2),
and are not duplicated here.

These tests are skipped rather than failed when the raw data is absent.
`research/data/raw/` is gitignored (`GITHUB_PLAN.md` Section 3), so a fresh clone
legitimately has no data until `download.py` has been run — a hard failure there
would mean CI could never be green for anyone who has not downloaded the datasets.
The manifest itself IS committed, so its absence is a real failure, not a skip.
"""

from __future__ import annotations

import pytest

from research.src.data.download import (
    MANIFEST_PATH,
    RAW_DIR,
    _existing_raw_files,
    read_manifest,
    sha256_of,
)

# Datasets that must appear in the manifest once acquired. Notri-Fact is absent
# until Kaggle credentials exist (DECISION_REGISTER.md M1-3's sibling blocker,
# reported at Milestone 1); tests covering it skip rather than fail so the suite
# stays honest about what has actually been acquired.
EXPECTED_AX_TO_GRIND_FILES = {
    "ax_to_grind/Fake News.csv",
    "ax_to_grind/True News.csv",
    "ax_to_grind/Combined .csv",
}


def test_manifest_exists() -> None:
    """The manifest is committed, so it must be present in any checkout."""
    assert MANIFEST_PATH.exists(), (
        f"{MANIFEST_PATH} is missing. It is a committed artifact "
        "(ROADMAP.md Milestone 1); regenerate it with `python -m research.src.data.download`."
    )


def test_manifest_is_parseable_and_non_empty() -> None:
    checksums = read_manifest()
    assert checksums, "MANIFEST.sha256 parsed to zero entries."
    for relative, checksum in checksums.items():
        assert len(checksum) == 64, f"{relative}: expected a 64-hex-char SHA256, got {checksum!r}"
        assert all(c in "0123456789abcdef" for c in checksum), (
            f"{relative}: checksum is not lowercase hex: {checksum!r}"
        )
        assert not relative.startswith("/") and "\\" not in relative, (
            f"{relative}: manifest paths must be relative and use forward slashes, "
            "so the file is identical across Windows and Linux."
        )


def test_manifest_covers_ax_to_grind() -> None:
    """All three acquired Ax-to-Grind files are recorded.

    `Combined .csv` is included deliberately — see DECISION_REGISTER.md M1-3: the
    upstream `Fake News.csv` has had its Urdu text destroyed, and `Combined .csv` is
    the only intact source of fake-labelled text.
    """
    recorded = set(read_manifest())
    missing = EXPECTED_AX_TO_GRIND_FILES - recorded
    assert not missing, f"Manifest is missing Ax-to-Grind entries: {sorted(missing)}"


@pytest.mark.parametrize("relative", sorted(EXPECTED_AX_TO_GRIND_FILES))
def test_file_present_and_checksum_matches(relative: str) -> None:
    """Every manifest entry must exist on disk and hash to its recorded value.

    This is the actual reproducibility anchor: it is what detects a truncated
    download, a partially-written file, or a raw file edited in place in violation
    of DATASET_PLAN.md Section 3's immutability rule.
    """
    checksums = read_manifest()
    if relative not in checksums:
        pytest.fail(f"{relative} is not recorded in MANIFEST.sha256")

    path = RAW_DIR / relative
    if not path.exists():
        pytest.skip(
            f"{relative} not present locally — research/data/raw/ is gitignored. "
            "Run `python -m research.src.data.download` to acquire it."
        )

    actual = sha256_of(path)
    assert actual == checksums[relative], (
        f"{relative} checksum mismatch.\n"
        f"  expected (manifest): {checksums[relative]}\n"
        f"  actual (on disk):    {actual}\n"
        "The file changed after download. raw/ is immutable "
        "(DATASET_PLAN.md Section 3) — re-run download.py rather than editing it."
    )


def test_no_unrecorded_files_in_raw() -> None:
    """Every file in raw/ is accounted for in the manifest.

    Guards the reverse direction of the checks above: a stray or hand-added file in
    raw/ would otherwise flow into the pipeline with no recorded provenance.
    """
    present = _existing_raw_files()
    if not present:
        pytest.skip("No raw data present locally; nothing to reconcile.")

    recorded = set(read_manifest())
    unrecorded = {p.relative_to(RAW_DIR).as_posix() for p in present} - recorded
    assert not unrecorded, (
        f"Files in research/data/raw/ are absent from MANIFEST.sha256: {sorted(unrecorded)}. "
        "Re-run download.py so the manifest covers everything present."
    )
