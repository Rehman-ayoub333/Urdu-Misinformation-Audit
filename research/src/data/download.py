"""Fetch the raw source datasets and record their SHA256 manifest.

Implements `DATASET_PLAN.md` Section 2 step 1. This is the only script permitted to
write into `research/data/raw/`; that directory is immutable afterwards
(`DATASET_PLAN.md` Section 3) — a correction is made by re-running this script
against a newly pinned source and recording a new manifest entry, never by editing
a raw file in place.

Sources (`DATASET_PLAN.md` Section 1):

- Ax-to-Grind Urdu (primary training set, 10,083 articles) — `Fake News.csv` and
  `True News.csv` from github.com/Sheetal83/Ax-to-Grind-Urdu-Dataset. Fetched over
  plain HTTPS from raw.githubusercontent.com, pinned to a commit SHA rather than
  `main`, so a later upstream push cannot silently change what this script returns.
- Notri-Fact Urdu (cross-dataset test set, 13,388 articles) — the Kaggle dataset
  `tridata/notri-fact-real-and-unreal-urdu-news`. Kaggle requires an API token;
  see `_download_notri_fact` for the credential contract.

UFND is deliberately absent: `DECISION_REGISTER.md` U1 records it as excluded after
its download link could not be located.

Usage:
    python -m research.src.data.download              # both datasets
    python -m research.src.data.download --only ax_to_grind
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import requests

# --- Source pinning -----------------------------------------------------------

# Pinned commit of github.com/Sheetal83/Ax-to-Grind-Urdu-Dataset (main @ 2024-02-19).
# REPRODUCIBILITY.md Section 3: raw inputs must stay identifiable for any past run.
AX_TO_GRIND_COMMIT = "217d3fa64aded3fd0553fb8f3e7e67eba8c3dba2"
AX_TO_GRIND_RAW_BASE = (
    f"https://raw.githubusercontent.com/Sheetal83/Ax-to-Grind-Urdu-Dataset/{AX_TO_GRIND_COMMIT}"
)
# Upstream filenames contain a space; they are kept verbatim so the manifest and the
# committed filenames match the source exactly.
#
# `Combined .csv` is fetched in addition to the two files named in DATASET_PLAN.md
# Section 1, because the upstream repository's contents do not match their names:
#
#   * `Fake News.csv`  — tab-separated, NOT valid UTF-8, and despite its name it holds
#                        the FULL dataset (5,053 FAKE + 5,030 TRUE). Every Urdu
#                        character has been replaced by a literal '?' by a lossy
#                        encoding conversion upstream: 2,694,909 of its 3,664,112
#                        bytes are 0x3F and only 27 Arabic-script characters survive.
#                        The text is irrecoverable from this file.
#   * `True News.csv`  — comma-separated, UTF-8 with BOM, intact. 5,030 TRUE rows,
#                        633,940 Arabic-script characters. This is genuinely the
#                        true-only half.
#   * `Combined .csv`  — comma-separated, UTF-8 with BOM, intact. The full dataset,
#                        5,053 FAKE + 5,030 TRUE = 10,083 rows, matching the size
#                        DATASET_PLAN.md documents.
#
# So the FAKE half's text exists ONLY in `Combined .csv`. All three are downloaded
# and checksummed: the two named files are kept as the provenance record of the
# corruption, and `Combined .csv` is the only usable source of fake-labelled text.
# Which file Milestone 2 actually reads is recorded in DECISION_REGISTER.md, not
# decided here.
AX_TO_GRIND_FILES = ("Fake News.csv", "True News.csv", "Combined .csv")

NOTRI_FACT_KAGGLE_REF = "tridata/notri-fact-real-and-unreal-urdu-news"

REQUEST_TIMEOUT_SECONDS = 60
CHUNK_BYTES = 1 << 16

# --- Paths --------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parents[3] / "research" / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "MANIFEST.sha256"

DATASETS = ("ax_to_grind", "notri_fact")


class DownloadError(RuntimeError):
    """Raised when a dataset cannot be acquired. Never swallowed into a partial run."""


def sha256_of(path: Path) -> str:
    """Streaming SHA256, so large CSVs are never fully read into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    """Stream `url` to `destination`, writing to a temp path first.

    The temp-then-rename avoids leaving a truncated file behind that a later run
    would happily checksum as if it were complete.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".partial")

    response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise DownloadError(f"GET {url} returned HTTP {response.status_code}")

    with temp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            if chunk:
                handle.write(chunk)

    temp_path.replace(destination)


def _download_ax_to_grind() -> list[Path]:
    """Fetch Ax-to-Grind's two label CSVs from the pinned commit."""
    target_dir = RAW_DIR / "ax_to_grind"
    written: list[Path] = []

    for filename in AX_TO_GRIND_FILES:
        # requests quotes the space in the path for us.
        url = f"{AX_TO_GRIND_RAW_BASE}/{requests.utils.quote(filename)}"
        destination = target_dir / filename
        print(f"[ax_to_grind] GET {url}")
        _download_file(url, destination)
        print(f"[ax_to_grind] wrote {destination.name} ({destination.stat().st_size:,} bytes)")
        written.append(destination)

    return written


def _download_notri_fact() -> list[Path]:
    """Fetch Notri-Fact from Kaggle.

    Kaggle's API requires credentials. With `kaggle==2.2.4` the accepted forms are
    (any one of):

      * `kaggle auth login` — OAuth web flow, caches credentials locally;
      * `KAGGLE_API_TOKEN` — API token from kaggle.com/settings/api, as an env var;
      * `~/.kaggle/access_token` — the same token, in a file.

    Note this is *not* the older `kaggle.json` / `KAGGLE_USERNAME` + `KAGGLE_KEY`
    scheme that most documentation still describes; 2.x replaced it.

    None of these appear in `DEPLOYMENT_PLAN.md` Section 6's env-var list, and that
    is correct rather than an omission: that list covers the deployed *product*,
    whereas this is a research-time-only credential that never reaches the backend,
    the frontend, or any deployed environment.

    The import is deliberately local to this function. `kaggle` 2.2.4 prints its help
    text and calls `sys.exit(1)` *at import time* when credentials are absent, so a
    module-level import would kill the Ax-to-Grind path too — which needs no
    credentials at all. `SystemExit` derives from `BaseException`, not `Exception`,
    so it has to be named explicitly here.
    """
    target_dir = RAW_DIR / "notri_fact"
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except (OSError, SystemExit) as exc:
        raise DownloadError(
            "Kaggle credentials not found. Notri-Fact cannot be downloaded.\n"
            "Authenticate with ONE of:\n"
            "  kaggle auth login                      (OAuth web flow)\n"
            "  export KAGGLE_API_TOKEN=<token>        (kaggle.com/settings/api)\n"
            "  write the token to ~/.kaggle/access_token\n"
            "then re-run this script. This dataset is not fetchable anonymously, and "
            "no substitute, sample, or placeholder is generated in its absence "
            "(CLAUDE.md rule 2).\n"
            f"Underlying error: {exc!r}"
        ) from exc

    try:
        api = KaggleApi()
        api.authenticate()
        print(f"[notri_fact] downloading Kaggle dataset {NOTRI_FACT_KAGGLE_REF}")
        api.dataset_download_files(NOTRI_FACT_KAGGLE_REF, path=str(target_dir), unzip=True)
    except SystemExit as exc:
        # Same reason as the import above: the client exits rather than raising.
        raise DownloadError(
            f"Kaggle API call failed (exit {exc.code}). Most often this is missing or "
            "expired credentials — see this function's docstring for the accepted forms."
        ) from exc

    # Kaggle occasionally leaves the archive behind even with unzip=True.
    for leftover in target_dir.glob("*.zip"):
        leftover.unlink()

    written = sorted(p for p in target_dir.rglob("*") if p.is_file())
    if not written:
        raise DownloadError(
            f"Kaggle reported success but {target_dir} is empty — refusing to record "
            "an empty manifest entry."
        )

    for path in written:
        print(f"[notri_fact] wrote {path.name} ({path.stat().st_size:,} bytes)")

    return written


def write_manifest(paths: Iterable[Path]) -> Path:
    """Write `research/data/raw/MANIFEST.sha256` in `sha256sum`-compatible format.

    Paths are recorded relative to `research/data/raw/` with forward slashes, so the
    manifest is identical whether it was generated on Windows or Linux. Entries are
    sorted for a stable diff.

    Per REPRODUCIBILITY.md Section 3 this manifest *is* the dataset version
    identifier. It is regenerated wholesale here rather than appended to; the
    "preserve the old entry" rule in that section applies to a deliberate upstream
    version change, which is a manual, reviewed edit — not something this script
    should guess at.
    """
    entries = []
    for path in paths:
        relative = path.relative_to(RAW_DIR).as_posix()
        entries.append(f"{sha256_of(path)}  {relative}")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text("\n".join(sorted(entries)) + "\n", encoding="utf-8")
    return MANIFEST_PATH


def read_manifest() -> dict[str, str]:
    """Parse MANIFEST.sha256 into {relative_path: sha256}."""
    if not MANIFEST_PATH.exists():
        return {}

    checksums: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        checksum, _, relative = line.partition("  ")
        checksums[relative] = checksum
    return checksums


def _existing_raw_files() -> list[Path]:
    """Every raw data file currently on disk, excluding the manifest itself."""
    if not RAW_DIR.exists():
        return []
    return sorted(
        p
        for p in RAW_DIR.rglob("*")
        if p.is_file() and p.name not in {"MANIFEST.sha256", ".gitkeep"}
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=DATASETS,
        help="Download a single dataset instead of all of them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target directory already has files.",
    )
    args = parser.parse_args(argv)

    requested = (args.only,) if args.only else DATASETS
    handlers = {
        "ax_to_grind": _download_ax_to_grind,
        "notri_fact": _download_notri_fact,
    }

    failures: dict[str, str] = {}

    for name in requested:
        target_dir = RAW_DIR / name
        already = [p for p in target_dir.glob("*") if p.is_file()] if target_dir.exists() else []
        if already and not args.force:
            print(f"[{name}] {len(already)} file(s) already present, skipping (use --force)")
            continue

        if args.force and target_dir.exists():
            shutil.rmtree(target_dir)

        try:
            handlers[name]()
        except DownloadError as exc:
            failures[name] = str(exc)
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)

    present = _existing_raw_files()
    if present:
        manifest = write_manifest(present)
        print(f"\nWrote {manifest} with {len(present)} entry/entries:")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            print(f"  {line}")
    else:
        print("\nNo raw files present; manifest not written.", file=sys.stderr)

    if failures:
        print(
            "\nIncomplete acquisition. The following datasets were NOT downloaded:",
            file=sys.stderr,
        )
        for name in failures:
            print(f"  - {name}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
