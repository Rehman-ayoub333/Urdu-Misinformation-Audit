"""Put metrics on the Hugging Face Hub as soon as they exist.

`DECISION_REGISTER.md` M4-6. Milestone 4's six training runs all completed and all
six checkpoints reached their staging repos — and the metrics were still lost,
because they only ever existed on a Kaggle draft session's working directory,
which did not survive the session ending. The zip was packaged; the download
silently failed; the directory was gone before it could be retried.

The bug was an asymmetry, not bad luck. Checkpoints were pushed **per seed**,
explicitly so "an interrupted session does not lose completed work". Metrics were
packaged **once, at the very end**. The cheap artefact — the one that is the actual
research output — had the weaker durability guarantee. This module removes the
asymmetry: metrics go to the Hub the moment they are written, seed by seed.

Results land in a **dataset** repo (`<prefix>/urdu-misinfo-results-staging`), not
in the model repos, because they are not model artefacts and are wanted
independently of any one checkpoint. Raw JSON files are uploaded individually
rather than only as a zip, so each is separately fetchable and diffable, and one
corrupt upload cannot take the rest with it.

**This never raises.** A failed upload must not destroy a finished training run —
that would be strictly worse than the problem it is solving. It retries, then
reports failure loudly in its return value, and the caller is expected to check.
`verify_results_uploaded()` is the check: run it at the end of a session and it
fails the run if anything is missing from the Hub.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Same credential contract as the checkpoint push (transformer.py) — environment
# only, never hardcoded, never committed.
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_STAGING_PREFIX = "HF_STAGING_PREFIX"

RESULTS_REPO_SUFFIX = "urdu-misinfo-results-staging"
REPO_TYPE = "dataset"

# Where files land inside the dataset repo. Keeps a future milestone's results from
# colliding with Milestone 4's.
DEFAULT_SUBDIR = "milestone4/metrics"


def results_repo_id(prefix: str | None = None) -> str:
    prefix = (prefix if prefix is not None else os.environ.get(ENV_HF_STAGING_PREFIX, "")).strip()
    return f"{prefix}/{RESULTS_REPO_SUFFIX}" if prefix else RESULTS_REPO_SUFFIX


def push_result_files(
    paths: Sequence[str | Path],
    *,
    subdir: str = DEFAULT_SUBDIR,
    dry_run: bool = False,
    attempts: int = 3,
) -> dict[str, Any]:
    """Upload metrics files to the results dataset repo. Never raises.

    Returns a status dict; `pushed` is False if anything failed, and `failed`
    names the files, so the caller can report or re-try. A dry run reports what it
    would have done and touches nothing — the same contract as `push_checkpoint`.
    """
    paths = [Path(p) for p in paths]
    prefix = os.environ.get(ENV_HF_STAGING_PREFIX, "").strip()
    token = os.environ.get(ENV_HF_TOKEN, "").strip()
    repo_id = results_repo_id(prefix)

    if dry_run:
        return {
            "pushed": False,
            "reason": "dry run",
            "would_push_to": repo_id,
            "would_push_files": [p.name for p in paths],
            "prefix_set": bool(prefix),
            "token_set": bool(token),
        }

    if not paths:
        return {"pushed": True, "repo_id": repo_id, "uploaded": [], "failed": []}

    if not prefix or not token:
        # Loud, but not fatal: the run's real output still exists locally, and
        # killing a finished training run over a missing env var would be worse.
        missing = [
            n for n, v in ((ENV_HF_STAGING_PREFIX, prefix), (ENV_HF_TOKEN, token)) if not v
        ]
        message = f"results NOT pushed: {missing} not set in the environment"
        print(f"    !! {message}")
        return {"pushed": False, "reason": message, "uploaded": [], "failed": [p.name for p in paths]}

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        message = f"results NOT pushed: could not create/access {repo_id}: {exc}"
        print(f"    !! {message}")
        return {"pushed": False, "reason": message, "uploaded": [], "failed": [p.name for p in paths]}

    uploaded: list[str] = []
    failed: list[str] = []

    for path in paths:
        if not path.exists():
            print(f"    !! results push: {path} does not exist, skipping")
            failed.append(path.name)
            continue

        target = f"{subdir}/{path.name}" if subdir else path.name
        for attempt in range(1, attempts + 1):
            try:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=target,
                    repo_id=repo_id,
                    repo_type=REPO_TYPE,
                )
                uploaded.append(target)
                break
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                if attempt == attempts:
                    print(f"    !! results push FAILED for {path.name} after {attempts} tries: {exc}")
                    failed.append(path.name)
                else:
                    # A transient 5xx from the Hub is common enough to be worth a
                    # short backoff rather than an immediate give-up.
                    time.sleep(2 * attempt)

    status = {
        "pushed": not failed,
        "repo_id": repo_id,
        "repo_type": REPO_TYPE,
        "uploaded": uploaded,
        "failed": failed,
    }
    if uploaded:
        print(f"    results -> {repo_id}/{subdir} ({len(uploaded)} file(s))")
    return status


def verify_results_uploaded(
    expected_filenames: Sequence[str],
    *,
    subdir: str = DEFAULT_SUBDIR,
) -> dict[str, Any]:
    """Confirm every expected metrics file is actually on the Hub.

    This is the check that makes M4-6's rule real: a run is not done until its
    results are off the ephemeral disk. Intended to be called at the end of a
    session and to FAIL the run if anything is missing — unlike the push itself,
    which is deliberately non-fatal.
    """
    prefix = os.environ.get(ENV_HF_STAGING_PREFIX, "").strip()
    token = os.environ.get(ENV_HF_TOKEN, "").strip()
    repo_id = results_repo_id(prefix)

    if not prefix or not token:
        return {"verified": False, "reason": "HF credentials not set", "missing": list(expected_filenames)}

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        listed = set(api.list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE))
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        return {"verified": False, "reason": f"could not list {repo_id}: {exc}", "missing": list(expected_filenames)}

    missing = [
        name
        for name in expected_filenames
        if (f"{subdir}/{name}" if subdir else name) not in listed
    ]
    return {
        "verified": not missing,
        "repo_id": repo_id,
        "n_expected": len(expected_filenames),
        "missing": missing,
    }
