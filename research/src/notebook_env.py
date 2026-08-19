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

**Detection is Kaggle-first, and that is load-bearing.** Kaggle's notebook image
is built `FROM` the Colab runtime image, so `import google.colab` SUCCEEDS on
Kaggle while being completely non-functional there. An earlier version of this
module treated that import as proof of Colab and checked it first, on the stated
but false assumption that "no environment provides both helper packages" — which
contradicted this project's own M4-3 finding. It misidentified every Kaggle
session. See `detect_platform()` for the full account.

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

# --------------------------------------------------------------------------
# Detection signals.
#
# ORDER IS LOAD-BEARING: Kaggle is checked first, and this is a correctness
# requirement, not a tie-break convention. See detect_platform().
#
# Env vars Kaggle sets on every notebook session. KAGGLE_KERNEL_RUN_TYPE is the
# one the notebook's own pre-flight cell already uses successfully.
_KAGGLE_ENV_VARS = (
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_URL_BASE",
    "KAGGLE_CONTAINER_NAME",
    "KAGGLE_DATA_PROXY_TOKEN",
)
# Kaggle's mount root. Module-level so tests can point it somewhere real.
_KAGGLE_DIR = "/kaggle"

# Env vars Colab sets. Unlike `google.colab`, these do NOT survive into Kaggle's
# image, so they are a genuine positive signal rather than an inherited artifact.
_COLAB_ENV_VARS = (
    "COLAB_RELEASE_TAG",
    "COLAB_GPU",
    "COLAB_JUPYTER_IP",
    "COLAB_BACKEND_VERSION",
)


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


def kaggle_signals() -> list[str]:
    """Positive evidence that this is a Kaggle notebook session."""
    found = [f"env:{name}" for name in _KAGGLE_ENV_VARS if os.environ.get(name)]
    if os.path.isdir(_KAGGLE_DIR):
        found.append(f"dir:{_KAGGLE_DIR}")
    # Last, and never alone-sufficient in practice: weaker than the others because
    # a module by this name could in principle be installed anywhere. It is listed
    # so the diagnostic is complete.
    if _can_import("kaggle_secrets"):
        found.append("import:kaggle_secrets")
    return found


def colab_signals() -> list[str]:
    """Positive evidence that this is a Colab session.

    `google.colab` is listed LAST and deliberately treated as the weakest signal:
    it is importable on Kaggle too, because Kaggle's notebook image is built
    `FROM` the Colab runtime image (`DECISION_REGISTER.md` M4-3) and inherits its
    site-packages. The COLAB_* environment variables do not survive that
    inheritance, so they are the trustworthy ones.
    """
    found = [f"env:{name}" for name in _COLAB_ENV_VARS if os.environ.get(name)]
    if _can_import("google.colab"):
        found.append("import:google.colab")
    return found


def platform_markers() -> dict[str, list[str]]:
    """Every detection signal found, for diagnostics.

    The notebook prints this next to the detected platform. When detection went
    wrong on a live Kaggle session on 2026-08-19 there was nothing to inspect —
    only the wrong answer — which is what turned a one-line bug into a round trip.
    """
    return {KAGGLE: kaggle_signals(), COLAB: colab_signals()}


def detect_platform() -> str:
    """Return COLAB, KAGGLE or UNKNOWN.

    **Kaggle is checked first, and that ordering is a correctness requirement.**

    The inheritance is one-directional: Kaggle's notebook image is built `FROM`
    the Colab runtime image (`DECISION_REGISTER.md` M4-3), so Colab's packages
    are present on Kaggle, while Kaggle's markers never appear on Colab. Checking
    Colab first therefore misidentifies every Kaggle session, which is exactly
    what happened live on 2026-08-19: `import google.colab` succeeded on a Kaggle
    P100, detection returned "colab", and `userdata.get()` hung and then raised
    Colab's own TimeoutException ("Secrets can only be fetched when running from
    the Colab UI") — a present-but-nonfunctional leftover, the same category of
    failure as the inherited torchvision copy M4-3 already documents.

    A Kaggle-positive signal is therefore conclusive; a Colab-positive one is
    only meaningful once Kaggle has been ruled out.
    """
    if kaggle_signals():
        return KAGGLE
    if colab_signals():
        return COLAB
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
                f"Could not read {name} from Colab Secrets (detected platform: "
                f"{platform}). Open the key icon in the left sidebar, add {name} "
                f"(HF token role: write), enable 'Notebook access' for this notebook, "
                f"then rerun this cell.\n\n"
                f"If you are NOT actually on Colab, this is a misdetection, not a "
                f"missing secret — the Colab secrets API raises this same timeout when "
                f"called outside the Colab UI. Signals found: {platform_markers()}. "
                f"Report that mapping rather than working around it.\n\n({exc})"
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
