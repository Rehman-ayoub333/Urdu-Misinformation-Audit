"""Environment gate for the Milestone 4 GPU run. Fails in seconds, not an hour in.

**Runs as a SUBPROCESS, deliberately, and that is the fix for `DECISION_REGISTER.md`
M4-5.** It used to be inline notebook-kernel code, which made it the only part of
this pipeline that could be poisoned by a stale kernel:

The hosted images import numpy at kernel boot. `pip install` then replaces numpy on
disk, but the kernel keeps the module objects it already has. numpy's `char` and
`_core.strings` submodules are LAZY, so the first `from numpy import *` after the
install loads those two files from the NEW on-disk numpy while `numpy._core.umath`
is still the OLD cached one — and the new `strings.py` asks for a `_center` ufunc
the old compiled extension does not have:

    ImportError: cannot import name '_center' from 'numpy._core.umath'

pip is right and the kernel is right at the same time, which is why the failure
reports the base image's numpy version while pip reports the pinned one.

Every real step of this pipeline already runs as its own `python -m ...` process
(`run_in_domain`, `transformer --dry-run`, the data checks), so it reads the
freshly installed packages and is unaffected. Running this gate the same way means
it validates *the environment the training actually uses*, rather than the
notebook kernel's stale view of it — no kernel restart required.

Exit code 0 = good to train. Non-zero = stop; the message says what to fix.
"""

from __future__ import annotations

import importlib.util
import sys

EXPECTED_PYTHON = "3.12"  # REPRODUCIBILITY.md Section 1 / DECISION_REGISTER.md M4-2
EXPECTED_TORCH = "2.13.0"  # research/requirements.txt
EXPECTED_TRANSFORMERS = "5.15.0"
EXPECTED_NUMPY = "2.5.2"
EXPECTED_SCIPY = "1.18.0"

failures: list[str] = []


def check(label: str, condition: bool, remedy: str) -> None:
    if condition:
        print(f"  OK   {label}")
    else:
        print(f"  FAIL {label}")
        failures.append(f"{label}\n       -> {remedy}")


print("Environment gate (fresh subprocess, not the notebook kernel)")
print("-" * 70)

# --- Interpreter ----------------------------------------------------------
python_version = ".".join(sys.version.split()[0].split(".")[:2])
check(
    f"python {sys.version.split()[0]} (expected {EXPECTED_PYTHON}.x)",
    python_version == EXPECTED_PYTHON,
    "The classical baselines were computed on 3.12; a different interpreter "
    "reintroduces the ambiguity M4-2 removed. Report this rather than working around it.",
)

# --- torchvision must be ABSENT (M4-3) ------------------------------------
check(
    "torchvision absent",
    importlib.util.find_spec("torchvision") is None,
    "Re-run the dependency cell (step 1 uninstalls it). A torchvision built against "
    "a different torch makes `from transformers import Trainer` raise RuntimeError.",
)

# --- numpy integrity (M4-4 symptom, M4-5 root cause) ----------------------
try:
    import numpy

    numpy_version = numpy.__version__
    # numpy.char / numpy._core.strings are lazy; this forces them, and it is the
    # cheap reproducer for a numpy whose .py files and compiled extension disagree.
    #
    # exec into a throwaway namespace, rather than a bare `from numpy import *`, so
    # ~600 numpy names do not shadow this script's own (`all`, `any`, `sum`...).
    # Walking numpy.__all__ with getattr was tried instead and is NOT equivalent: it
    # reaches numpy.testing first and raises a different exception, so it does not
    # reproduce what scipy's shim actually does.
    exec("from numpy import *", {})  # noqa: S102 - faithful reproducer, fixed input
    numpy_ok = True
    numpy_error = ""
except (ImportError, AttributeError) as exc:
    # AttributeError as well as ImportError: a mismatched numpy raises either,
    # depending on which submodule is reached first (observed both while testing).
    numpy_version = locals().get("numpy_version", "?")
    numpy_ok = False
    numpy_error = f"{type(exc).__name__}: {exc}"

check(
    f"numpy {numpy_version} imports completely (char/strings)",
    numpy_ok,
    f"ImportError: {numpy_error}\n"
    "          In a subprocess this means the ON-DISK install is genuinely broken "
    "(not merely a stale kernel). Re-run the dependency cell, which force-reinstalls "
    "numpy and scipy.",
)
check(
    f"numpy == {EXPECTED_NUMPY}",
    numpy_version == EXPECTED_NUMPY,
    f"Found {numpy_version}. The pinned install did not take effect on disk; "
    "re-run the dependency cell.",
)

# --- scipy is REQUIRED here (M4-4): sklearn.metrics pulls in ~493 modules --
try:
    import scipy

    scipy_version = scipy.__version__
    scipy_ok = True
except Exception as exc:  # noqa: BLE001 - report, never abort the gate
    scipy_version = "missing"
    scipy_ok = False
    print(f"       (scipy import error: {exc})")

check(
    f"scipy {scipy_version} imports",
    scipy_ok,
    "scipy is NOT optional here: sklearn.metrics, which every metrics file goes "
    "through, imports it. Re-run the dependency cell.",
)
check(
    f"scipy == {EXPECTED_SCIPY}",
    scipy_version == EXPECTED_SCIPY,
    f"Found {scipy_version}; re-run the dependency cell.",
)

# --- the import that actually failed on the first Colab attempt -----------
try:
    import torch
    import transformers
    from transformers import Trainer  # noqa: F401

    trainer_ok = True
    trainer_error = ""
except Exception as exc:  # noqa: BLE001 - this is the failure being gated on
    trainer_ok = False
    trainer_error = f"{type(exc).__name__}: {exc}"

check(
    "from transformers import Trainer",
    trainer_ok,
    f"{trainer_error}\n          Re-run the dependency cell and read the errors it prints.",
)

if trainer_ok:
    # torch appends a build tag: "2.13.0+cu130" on GPU, "2.13.0+cpu" locally. Same
    # pinned version, different accelerator, so compare the release part only.
    torch_version = torch.__version__.split("+")[0]
    check(
        f"torch {torch.__version__} == {EXPECTED_TORCH}",
        torch_version == EXPECTED_TORCH,
        "The pinned install did not take effect; re-run the dependency cell.",
    )
    check(
        f"transformers {transformers.__version__} == {EXPECTED_TRANSFORMERS}",
        transformers.__version__ == EXPECTED_TRANSFORMERS,
        "The pinned install did not take effect; re-run the dependency cell.",
    )
    check(
        "torch sees a GPU",
        torch.cuda.is_available(),
        "Check the accelerator setting (Kaggle: Settings -> Accelerator; "
        "Colab: Runtime -> Change runtime type).",
    )
    if torch.cuda.is_available():
        print(f"       device: {torch.cuda.get_device_name(0)}")
        print(f"       visible GPUs: {torch.cuda.device_count()} (1 expected, M4-3)")

print("-" * 70)
if failures:
    print(f"\n{len(failures)} CHECK(S) FAILED — do not start training:\n")
    for i, failure in enumerate(failures, 1):
        print(f"  {i}. {failure}\n")
    sys.exit(1)

print("All environment checks passed. Safe to run the smoke test.")
