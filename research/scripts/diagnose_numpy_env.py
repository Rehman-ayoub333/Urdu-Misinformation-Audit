"""Diagnose which numpy the kernel is actually using, and why.

TEMPORARY DIAGNOSTIC — not part of the pipeline, not imported by anything. Added
for `DECISION_REGISTER.md` M4-5 and kept because the same question ("is this a
second install, or a stale import?") will come up again on a hosted image.

Run it as a notebook cell (`%run research/scripts/diagnose_numpy_env.py`) or as a
subprocess (`!python research/scripts/diagnose_numpy_env.py`) — but note that only
the FIRST form can see the kernel's own module state, which is the whole point.

Deliberately read-only: it installs nothing, uninstalls nothing, and never runs
`from numpy import *` before it has reported, so a broken environment still
produces a full report instead of an exception.

The discriminator it exists to settle:

  * TWO INSTALLS  -> `find` reports numpy in more than one directory, and the
                     directory numpy resolves from is not the one pip targets.
  * STALE IMPORT  -> exactly ONE install, pip reports the pinned version, but the
                     kernel's already-imported numpy reports an older one, and a
                     fresh subprocess agrees with pip rather than with the kernel.
"""

from __future__ import annotations

import glob
import os
import site
import subprocess
import sys

SEP = "=" * 78


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


section("1. Was numpy ALREADY imported before this cell ran?")
# The single most important line in this report. If numpy is already in
# sys.modules, anything pip did to it on disk afterwards is NOT reflected in the
# module objects this kernel is holding.
already = "numpy" in sys.modules
print("numpy already in sys.modules :", already)
if already:
    print("  ^ if True, this kernel is holding a numpy imported BEFORE the pip cell")
loaded = sorted(m for m in sys.modules if m == "numpy" or m.startswith("numpy."))
print("numpy submodules loaded      :", len(loaded))
print("  lazy ones that matter here:")
for name in ("numpy.char", "numpy._core.strings", "numpy._core.defchararray", "numpy._core.umath"):
    print(f"    {name:<28} loaded={name in sys.modules}")

section("2. What does the KERNEL think numpy is?")
try:
    import numpy

    print("numpy.__version__ :", numpy.__version__)
    print("numpy.__file__    :", numpy.__file__)
    print("numpy.__path__    :", list(numpy.__path__))
    import numpy._core.umath as _umath

    print("umath.__file__    :", _umath.__file__)
    print("_center present   :", hasattr(_umath, "_center"))
    print("  ^ False here + newer numpy on disk = the stale-import failure mode")
except Exception as exc:  # noqa: BLE001 - diagnostic must never abort
    print("import numpy FAILED:", type(exc).__name__, exc)

section("3. What does a FRESH interpreter see? (subprocess, no cached modules)")
# If this disagrees with section 2, the on-disk install is fine and the KERNEL is
# stale. If it agrees, the problem really is on disk.
probe = (
    "import numpy, sys;"
    "print('version :', numpy.__version__);"
    "print('file    :', numpy.__file__);"
    "import numpy._core.umath as u;"
    "print('_center :', hasattr(u, '_center'));"
    "exec('from numpy import *');"
    "print('from numpy import * : OK')"
)
result = subprocess.run(
    [sys.executable, "-c", probe], capture_output=True, text=True, check=False
)
print(result.stdout.strip() or "(no stdout)")
if result.returncode != 0:
    print("subprocess FAILED:")
    print(result.stderr.strip()[-1500:])

section("4. How many numpy installs are on this filesystem?")
# The shadow-install hypothesis. More than one hit here (outside the same tree)
# means --force-reinstall touched a directory that is not the one being imported.
candidates: list[str] = []

# Every directory this interpreter would actually search, plus the usual hosted-image
# locations. Derived from the interpreter rather than hardcoded, so it is correct on
# any platform instead of only on Linux.
for base in (*sys.path, *site.getsitepackages(), site.getusersitepackages()):
    if base and os.path.isdir(base):
        candidates.append(os.path.join(base, "numpy", "_core", "umath.py"))

extra_globs = [
    "/usr/local/lib/python3*/dist-packages/numpy/_core/umath.py",
    "/usr/local/lib/python3*/site-packages/numpy/_core/umath.py",
    "/usr/lib/python3*/dist-packages/numpy/_core/umath.py",
    "/opt/conda/lib/python3*/site-packages/numpy/_core/umath.py",
    "/root/.local/lib/python3*/site-packages/numpy/_core/umath.py",
]
for pattern in extra_globs:
    candidates.extend(glob.glob(pattern))

found = sorted({os.path.realpath(c) for c in candidates if os.path.isfile(c)})
for hit in found:
    # Read the sibling version.py so each install is identified, not just located —
    # two installs of the SAME version would be harmless; differing ones would not.
    version = "?"
    version_file = os.path.join(os.path.dirname(os.path.dirname(hit)), "version.py")
    try:
        with open(version_file, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("version ="):
                    version = line.split("=", 1)[1].strip().strip("\"'")
                    break
    except OSError:
        pass
    print(f"  numpy {version:<10} {hit}")

print(f"total distinct numpy installs found: {len(found)}")
print("  1 = single install, so a stale import is the only explanation left")
print("  0 = none of the probed locations matched (expected off-Linux)")
print("  2+ = shadow install; --force-reinstall may have targeted the wrong one")

section("5. sys.path, in resolution order")
for i, entry in enumerate(sys.path):
    print(f"  [{i:>2}] {entry!r}")

section("6. What does pip believe is installed?")
for package in ("numpy", "scipy"):
    out = subprocess.run(
        [sys.executable, "-m", "pip", "show", "-f", package],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        print(f"  {package}: pip show unavailable ({out.stderr.strip()[:120] or 'no output'})")
        # Fall back to the interpreter's own metadata, which needs no pip.
        try:
            import importlib.metadata as md

            dist = md.distribution(package)
            print(f"    importlib.metadata says: {package} {dist.version}")
            print(f"    location: {dist.locate_file('')}")
        except Exception as exc:  # noqa: BLE001 - diagnostic must never abort
            print(f"    importlib.metadata also failed: {exc}")
    else:
        for line in out.stdout.splitlines():
            if line.startswith(("Name:", "Version:", "Location:")):
                print(" ", line)
    print()

section("VERDICT")
print(
    "Compare section 2 (kernel) against sections 3 and 6 (disk/pip):\n"
    "  * kernel older than disk, and section 4 shows ONE install\n"
    "        -> STALE IMPORT. pip worked; the kernel must be RESTARTED.\n"
    "  * section 4 shows TWO installs in different trees\n"
    "        -> SHADOW INSTALL. pip targeted the wrong one.\n"
    "  * kernel and subprocess agree, both broken\n"
    "        -> genuinely corrupt on-disk install."
)
