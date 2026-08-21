"""What files a pushed checkpoint must carry to count as complete.

Extracted so there is exactly one definition of "this branch holds a usable
checkpoint". Two callers need it and they must not drift:

* `research/scripts/inventory_staging_checkpoints.py`, which audits the staging
  repos before GPU time is spent (`DECISION_REGISTER.md` M4-6);
* `research/src/models/transformer.py`, which verifies a push before deleting the
  local copy — where disagreeing with the auditor would mean deleting a
  checkpoint the auditor would later call incomplete.

**Standard library only, deliberately.** The inventory script keeps its dependency
surface to `huggingface_hub` + `PyYAML` on purpose (it is a cheap pre-flight, run
before the heavy environment exists), so anything it imports from this package
tree has to be free of numpy, pandas and torch.

A branch needs all three groups. A branch with weights but no tokenizer is not
re-scorable, which is exactly the state M4-6's recovery had to rule out.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "CONFIG_FILES",
    "TOKENIZER_FILES",
    "WEIGHT_FILES",
    "first_present",
    "missing_artifact_kinds",
]

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
CONFIG_FILES = ("config.json",)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",  # XLM-R
    "vocab.txt",  # mBERT
    "spiece.model",
)


def first_present(present: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate filename that appears in `present`."""
    available = set(present)
    for name in candidates:
        if name in available:
            return name
    return None


def missing_artifact_kinds(present: Iterable[str]) -> list[str]:
    """Return the artifact groups absent from a file listing; empty means complete.

    Ordered config → weights → tokenizer so the message reads the same way
    everywhere it is printed.
    """
    available = set(present)
    return [
        kind
        for kind, candidates in (
            ("config", CONFIG_FILES),
            ("weights", WEIGHT_FILES),
            ("tokenizer", TOKENIZER_FILES),
        )
        if first_present(available, candidates) is None
    ]
