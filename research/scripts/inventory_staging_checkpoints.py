"""Step 0 of the Milestone 4 metrics recovery: what is actually on the Hub?

`DECISION_REGISTER.md` M4-6. The Kaggle draft session that produced Milestone 4's
six runs lost its working directory before the metrics zip could be downloaded, so
the metrics JSONs are gone. The CHECKPOINTS were pushed to the HF staging repos as
each seed finished and are believed intact — but only two of the six pushes were
confirmed by a printed dict that anyone actually read.

This script settles that before a single second of GPU time is spent: it lists both
staging repos, checks every seed branch, and verifies each one carries a model
config, model weights AND tokenizer files — all three are needed to re-score a
checkpoint, and a branch with weights but no tokenizer is not recoverable.

Deliberately cheap and narrow: CPU only, no torch, no sklearn, no pandas. It reads
the repo suffixes and seed list straight from the YAML configs so it cannot drift
from what the training run actually pushed to, and it needs nothing from this
project's own package tree beyond those files.

    python research/scripts/inventory_staging_checkpoints.py

Exit 0 = all six branches usable, go ahead with the eval-only recovery.
Exit 1 = at least one branch is missing or incomplete; the report names which, and
         only those seeds need retraining.
"""

from __future__ import annotations

import os
import pathlib
import sys

import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = _ROOT / "research" / "configs"

# Experiment id -> config file. Mirrors transformer.py's MODEL_CONFIGS, but read
# from disk here rather than imported, to keep this script's dependency surface to
# huggingface_hub + PyYAML.
MODEL_CONFIGS = {"C": "model_mbert.yaml", "D": "model_xlmr.yaml"}

ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_STAGING_PREFIX = "HF_STAGING_PREFIX"

# A branch must carry all three groups to be re-scorable.
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
CONFIG_FILES = ("config.json",)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",  # XLM-R
    "vocab.txt",  # mBERT
    "spiece.model",
)


def _first_present(present: set[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in present:
            return name
    return None


def main() -> int:
    prefix = os.environ.get(ENV_HF_STAGING_PREFIX, "").strip()
    token = os.environ.get(ENV_HF_TOKEN, "").strip()
    if not prefix or not token:
        missing = [
            n
            for n, v in ((ENV_HF_STAGING_PREFIX, prefix), (ENV_HF_TOKEN, token))
            if not v
        ]
        print(f"ERROR: {missing} not set. Run the secret cell first.")
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print("Milestone 4 staging-checkpoint inventory")
    print(f"namespace: {prefix}")
    print("=" * 78)

    problems: list[str] = []
    usable: list[tuple[str, int]] = []

    for experiment_id, config_name in MODEL_CONFIGS.items():
        config = yaml.safe_load((CONFIG_DIR / config_name).read_text(encoding="utf-8"))
        suffix = config["checkpoint_push"]["staging_repo_suffix"]
        repo_id = f"{prefix}/{suffix}"
        seeds = list(config["training"]["seeds"])

        print(f"\n{experiment_id} — {config['model']['name']}")
        print(f"  repo: {repo_id}")

        try:
            api.repo_info(repo_id=repo_id)
        except Exception as exc:  # noqa: BLE001 - report every repo, never abort early
            print(f"  REPO NOT FOUND OR UNREADABLE: {exc}")
            problems.extend(f"{experiment_id}/seed-{seed}: repo unreachable" for seed in seeds)
            continue

        for seed in seeds:
            branch = f"seed-{seed}"
            try:
                info = api.repo_info(repo_id=repo_id, revision=branch, files_metadata=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  seed-{seed:<5} MISSING BRANCH ({type(exc).__name__})")
                problems.append(f"{experiment_id}/seed-{seed}: branch missing")
                continue

            present = {s.rfilename for s in info.siblings}
            weights = _first_present(present, WEIGHT_FILES)
            cfg = _first_present(present, CONFIG_FILES)
            tok = _first_present(present, TOKENIZER_FILES)

            size_mb = None
            if weights:
                for sibling in info.siblings:
                    if sibling.rfilename == weights and getattr(sibling, "size", None):
                        size_mb = round(sibling.size / 1024**2, 1)

            missing_kinds = [
                kind
                for kind, found in (("config", cfg), ("weights", weights), ("tokenizer", tok))
                if not found
            ]

            if missing_kinds:
                print(f"  seed-{seed:<5} INCOMPLETE — missing {missing_kinds}")
                print(f"             files present: {sorted(present)}")
                problems.append(f"{experiment_id}/seed-{seed}: missing {missing_kinds}")
            else:
                sha = (getattr(info, "sha", None) or "?")[:12]
                size = f"{size_mb} MB" if size_mb else "size unknown"
                print(f"  seed-{seed:<5} OK   sha={sha}  {weights} ({size})  tokenizer={tok}")
                usable.append((experiment_id, seed))

    print("\n" + "=" * 78)
    print(f"usable checkpoints: {len(usable)}/6")

    if problems:
        print("\nPROBLEMS — these seeds are NOT recoverable by re-scoring:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nOnly the seeds listed above need retraining; every OK seed above can be "
            "re-scored without a training run. Retrain them with, for example:\n"
            "    python -m research.src.experiments.run_in_domain --models transformer "
            "--experiments <ID>"
        )
        return 1

    print("\nAll six checkpoints are present and complete.")
    print("Safe to run the eval-only recovery:")
    print("    python -m research.src.models.evaluate_checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
