"""Platform detection for the hosted-GPU notebooks (Colab and Kaggle).

Milestone 4's training run is a hosted-GPU handoff (`PROJECT_SPECIFICATION.md`
Section 6). It was written for Colab; Kaggle became the target when Colab's free
GPU quota ran out mid-milestone. The two platforms differ in exactly three
mechanical ways that the notebook touches — how a secret is read, where the
writable working directory lives, and how a produced file is handed back — and in
nothing else that matters here. Both now run Python 3.12 on the same underlying
Colab runtime image (Kaggle's Dockerfile builds `FROM` it), so the dependency
pins, the CUDA build and the training code are unchanged across them.

This module holds those three differences, and only those, so the notebook stays
ONE document rather than forking into a per-platform copy that would drift. It is
deliberately:

* **stdlib-only** — it is imported immediately after the repo is cloned and
  *before* `pip install -r research/requirements.txt` has run, so it cannot
  depend on anything from that file;
* **side-effect-free on import** — detection happens when a function is called,
  never at import time, so importing it on a laptop for a unit test is safe;
* **explicit on failure** — an unknown platform raises with the actual fix, not a
  bare KeyError three cells later.

Nothing here is research logic. `REPRODUCIBILITY.md` Section 7 forbids notebooks
carrying independent logic that produces a reported number; this produces none —
it resolves an environment. It lives in `research/src/` rather than inline in the
notebook precisely so it can be unit-tested (`research/tests/test_notebook_env.py`).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

COLAB = "colab"
KAGGLE = "kaggle"
UNKNOWN = "unknown"

# Writable working root per platform. Colab's is /content; Kaggle's is
# /kaggle/working, which is ALSO the only directory Kaggle persists and exposes
# in the notebook's Output tab — writing the metrics zip anywhere else on Kaggle
# produces a file nobody can retrieve.
_WORKING_ROOT = {COLAB: "/content", KAGGLE: "/kaggle/working"}


def _can_import(module: str) -> bool:
    """True if `module` is importable, without keeping it imported.

    `importlib.util.find_spec` is not enough on its own: Kaggle's and Colab's
    helper packages are both present-and-importable only inside their own
    runtime, and find_spec can succeed for a package whose import then fails.
    Actually importing is the honest check, and both modules are cheap.
    """
    try:
        importlib.import_module(module)
    except Exception:  # noqa: BLE001 - a failed import may raise anything at all
        return False
    return True


def detect_platform() -> str:
    """Return COLAB, KAGGLE or UNKNOWN.

    Order matters only for determinism, not correctness — no environment
    provides both helper packages. Colab is checked first because it is the
    platform this notebook was originally written for.
    """
    if _can_import("google.colab"):
        return COLAB
    # Kaggle sets this on every notebook session; the kaggle_secrets import is the
    # confirming check. Either alone is weaker: the env var is settable by hand,
    # and `kaggle_secrets` can be pip-installed locally.
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or _can_import("kaggle_secrets"):
        return KAGGLE
    return UNKNOWN


def get_secret(name: str) -> str:
    """Read `name` from the host platform's secret store.

    Colab  → Secrets panel (🔑 in the left sidebar), "Notebook access" enabled.
    Kaggle → Add-ons → Secrets, attached to this notebook.

    Falls back to the process environment when running on neither, so a local
    dry run can export the variable normally instead of needing a hosted runtime.

    Raises RuntimeError with the platform-appropriate fix on failure — never
    returns an empty string, because an empty HF_TOKEN fails an hour later at
    checkpoint-push time rather than here.
    """
    platform = detect_platform()

    if platform == COLAB:
        from google.colab import userdata  # type: ignore[import-not-found]

        try:
            value = userdata.get(name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read {name} from Colab Secrets. Open the key icon in the "
                f"left sidebar, add {name} (HF token role: write), enable 'Notebook "
                f"access' for this notebook, then rerun this cell. ({exc})"
            ) from exc

    elif platform == KAGGLE:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        try:
            value = UserSecretsClient().get_secret(name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read {name} from Kaggle Secrets. Open Add-ons -> Secrets, "
                f"add {name} (HF token role: write), tick it to attach it to THIS "
                f"notebook, then rerun this cell. Note that a secret added while the "
                f"notebook is running is not visible until you rerun the cell. ({exc})"
            ) from exc

    else:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(
                f"{name} is not available: this is neither a Colab nor a Kaggle "
                f"runtime (no google.colab, no kaggle_secrets), and {name} is not set "
                f"in the environment. Run this notebook on Colab or Kaggle, or export "
                f"{name} before starting a local session."
            )

    if not value:
        raise RuntimeError(
            f"{name} resolved to an empty value on {platform}. The secret exists but "
            f"has no content, or is not attached to this notebook."
        )
    return value


def working_root() -> Path:
    """Writable scratch root for this platform.

    Falls back to the current directory off-platform so a local run does not try
    to create /content or /kaggle/working at the filesystem root.
    """
    root = _WORKING_ROOT.get(detect_platform())
    return Path(root) if root else Path.cwd()


def repo_dir(name: str = "repo") -> Path:
    """Where the git checkout of this project should live on this platform.

    Kaggle's root filesystem is small and not persisted; /kaggle/working is the
    one place with real space. Colab's /content is the equivalent. Hardcoding
    /content — as this notebook originally did — puts the clone outside Kaggle's
    working directory, which is the sort of thing that works right up until the
    disk fills.
    """
    return working_root() / name


def deliver_file(path: str | Path) -> str:
    """Hand a produced file back to the human running the notebook.

    Colab can trigger a browser download directly. Kaggle cannot: anything under
    /kaggle/working is collected into the session's Output tab when the notebook
    finishes, and is downloaded from there. Returns a human-readable description
    of what happened, for the notebook to print.
    """
    path = Path(path)
    platform = detect_platform()

    if platform == COLAB:
        from google.colab import files  # type: ignore[import-not-found]

        try:
            files.download(str(path))
        except Exception as exc:  # noqa: BLE001 - never abort after an hour of training
            return f"Automatic download failed ({exc}). Get {path} from the Files pane."
        return f"Downloading {path.name} via the browser."

    if platform == KAGGLE:
        return (
            f"{path} is in Kaggle's working directory. Download it from the 'Output' "
            "tab of this notebook's session (right panel), or via 'Save Version' -> "
            "the version's Output tab. Kaggle has no direct browser-download call."
        )

    return f"{path} is on the local filesystem; no download step needed."
