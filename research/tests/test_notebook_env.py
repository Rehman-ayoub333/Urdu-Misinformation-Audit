"""Tests for the Colab/Kaggle platform detection used by the Milestone 4 notebook.

The real Kaggle secrets path cannot be exercised here — it needs a live Kaggle
session with an attached secret, which is Rehman's next step. What CAN be checked
without one, and is what these tests cover, is that the *branching* is correct:
that each platform is detected from the right signal, that each branch calls the
right secret API, that every failure mode raises a RuntimeError carrying an
actionable message rather than surfacing a bare KeyError or returning "", and
that the paths differ per platform in the way the notebook depends on.

Both platform packages are absent on the machine this test suite runs on, so both
are injected into sys.modules as stubs. That is the point: it verifies the code
picks the correct branch and calls the correct API on it, which is the part that
can silently be wrong.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from research.src import notebook_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Neither platform present, unless a test says otherwise.

    Clears every signal detect_platform() consults — both platforms' env vars, both
    helper modules, and the /kaggle directory probe (pointed at a path that does
    not exist, so a real /kaggle on the host cannot leak into a test).
    """
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "kaggle_secrets", raising=False)
    for name in notebook_env._KAGGLE_ENV_VARS + notebook_env._COLAB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(notebook_env, "_KAGGLE_DIR", str(tmp_path / "no-kaggle-here"))


def _install_colab(monkeypatch, *, secret="colab-token", raises=None, downloads=None):
    """Inject a fake `google.colab` exposing the two APIs the notebook uses."""
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    userdata = types.ModuleType("google.colab.userdata")
    files = types.ModuleType("google.colab.files")

    def get(name):
        if raises is not None:
            raise raises
        return secret

    userdata.get = get
    files.download = lambda p: (downloads.append(p) if downloads is not None else None)
    colab.userdata = userdata
    colab.files = files
    google.colab = colab

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _install_kaggle(monkeypatch, *, secret="kaggle-token", raises=None, seen=None):
    """Inject a fake `kaggle_secrets` exposing UserSecretsClient().get_secret()."""
    module = types.ModuleType("kaggle_secrets")

    class UserSecretsClient:
        def get_secret(self, name):
            if seen is not None:
                seen.append(name)
            if raises is not None:
                raise raises
            return secret

    module.UserSecretsClient = UserSecretsClient
    monkeypatch.setitem(sys.modules, "kaggle_secrets", module)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")


# --------------------------------------------------------------------------
# detect_platform
# --------------------------------------------------------------------------


def test_detects_colab(monkeypatch):
    _install_colab(monkeypatch)
    assert notebook_env.detect_platform() == notebook_env.COLAB


def test_detects_kaggle(monkeypatch):
    _install_kaggle(monkeypatch)
    assert notebook_env.detect_platform() == notebook_env.KAGGLE


def test_detects_kaggle_from_env_var_alone(monkeypatch):
    """Kaggle is recognised even if kaggle_secrets cannot be imported."""
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert notebook_env.detect_platform() == notebook_env.KAGGLE


def test_detects_neither(monkeypatch):
    assert notebook_env.detect_platform() == notebook_env.UNKNOWN


# --------------------------------------------------------------------------
# get_secret — the branch that actually matters
# --------------------------------------------------------------------------


def test_secret_from_colab(monkeypatch):
    _install_colab(monkeypatch, secret="hf_colab")
    assert notebook_env.get_secret("HF_TOKEN") == "hf_colab"


def test_secret_from_kaggle(monkeypatch):
    seen: list[str] = []
    _install_kaggle(monkeypatch, secret="hf_kaggle", seen=seen)
    assert notebook_env.get_secret("HF_TOKEN") == "hf_kaggle"
    # The requested name is forwarded verbatim, not hardcoded inside the branch.
    assert seen == ["HF_TOKEN"]


def test_kaggle_is_used_when_colab_absent(monkeypatch):
    """The actual fallback: no google.colab, so Kaggle's client is what runs."""
    _install_kaggle(monkeypatch, secret="fallback-worked")
    assert "google.colab" not in sys.modules
    assert notebook_env.get_secret("HF_TOKEN") == "fallback-worked"


# --------------------------------------------------------------------------
# REGRESSION: the live 2026-08-19 Kaggle failure.
#
# Kaggle's notebook image is built FROM the Colab runtime image, so google.colab
# is importable there but non-functional. Detection used to check Colab first and
# so returned "colab" on a real Kaggle P100; userdata.get() then hung and raised
# Colab's TimeoutException. The previous version of this file asserted the WRONG
# precedence ("colab wins when both present"), which is why 19 green tests did not
# catch it. These reproduce the ambiguous condition exactly.
# --------------------------------------------------------------------------


class _ColabTimeout(Exception):
    """Stand-in for google.colab.errors.NotebookAccessError / TimeoutException."""


def _simulate_kaggle_with_inherited_colab(monkeypatch, **kwargs):
    """A real Kaggle session: Kaggle's markers AND Colab's inherited package.

    The Colab stub raises on userdata.get(), exactly as the live session did — so
    a test that resolves to Colab fails loudly rather than quietly returning a
    plausible value.
    """
    _install_colab(
        monkeypatch,
        raises=_ColabTimeout(
            "Requesting secret HF_TOKEN timed out. Secrets can only be fetched "
            "when running from the Colab UI"
        ),
    )
    _install_kaggle(monkeypatch, **kwargs)


def test_kaggle_wins_over_inherited_colab_package(monkeypatch):
    """The exact live failure: both markers present must resolve to Kaggle."""
    _simulate_kaggle_with_inherited_colab(monkeypatch)
    assert notebook_env.detect_platform() == notebook_env.KAGGLE


def test_secret_read_survives_inherited_colab_package(monkeypatch):
    """End to end: the secret comes from Kaggle, not from the Colab timeout path."""
    _simulate_kaggle_with_inherited_colab(monkeypatch, secret="hf_from_kaggle")
    assert notebook_env.get_secret("HF_TOKEN") == "hf_from_kaggle"


def test_kaggle_dir_alone_beats_importable_google_colab(monkeypatch, tmp_path):
    """Even with no KAGGLE_* env var set, /kaggle outranks an importable google.colab."""
    _install_colab(monkeypatch, raises=_ColabTimeout("timed out"))
    kaggle_dir = tmp_path / "kaggle"
    kaggle_dir.mkdir()
    monkeypatch.setattr(notebook_env, "_KAGGLE_DIR", str(kaggle_dir))
    assert notebook_env.detect_platform() == notebook_env.KAGGLE


def test_paths_and_delivery_follow_the_corrected_detection(monkeypatch):
    """The misdetection also silently broke these two — /content on Kaggle is
    unretrievable, and files.download does not exist there."""
    _simulate_kaggle_with_inherited_colab(monkeypatch)
    assert notebook_env.working_root() == Path("/kaggle/working")
    assert notebook_env.repo_dir() == Path("/kaggle/working/repo")
    assert "Output" in notebook_env.deliver_file("/kaggle/working/metrics.zip")


def test_colab_still_detected_when_kaggle_is_genuinely_absent(monkeypatch):
    """The fix must not break Colab: no Kaggle marker means google.colab is trusted."""
    _install_colab(monkeypatch, secret="from-colab")
    assert notebook_env.detect_platform() == notebook_env.COLAB
    assert notebook_env.get_secret("HF_TOKEN") == "from-colab"


def test_colab_env_var_alone_detects_colab(monkeypatch):
    """COLAB_* env vars are the trustworthy Colab signal; they do not reach Kaggle."""
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab_20260514")
    assert notebook_env.detect_platform() == notebook_env.COLAB


def test_markers_report_both_sides(monkeypatch):
    """The diagnostic must show the evidence, which is what was missing live."""
    _simulate_kaggle_with_inherited_colab(monkeypatch)
    markers = notebook_env.platform_markers()
    assert "env:KAGGLE_KERNEL_RUN_TYPE" in markers[notebook_env.KAGGLE]
    assert "import:google.colab" in markers[notebook_env.COLAB]


def test_colab_branch_error_names_the_detected_platform(monkeypatch):
    """A Colab-path failure must say which platform was detected and why.

    The old message said 'Colab UI' unconditionally, which on Kaggle pointed at a
    UI that does not exist and hid the real cause.
    """
    _install_colab(monkeypatch, raises=_ColabTimeout("timed out"))
    with pytest.raises(RuntimeError) as excinfo:
        notebook_env.get_secret("HF_TOKEN")
    message = str(excinfo.value)
    assert "detected platform: colab" in message
    assert "misdetection" in message
    assert "import:google.colab" in message


def test_missing_secret_on_colab_raises_actionable_error(monkeypatch):
    _install_colab(monkeypatch, raises=KeyError("HF_TOKEN"))
    with pytest.raises(RuntimeError, match="Colab Secrets"):
        notebook_env.get_secret("HF_TOKEN")


def test_missing_secret_on_kaggle_raises_actionable_error(monkeypatch):
    _install_kaggle(monkeypatch, raises=KeyError("HF_TOKEN"))
    with pytest.raises(RuntimeError, match="Add-ons -> Secrets"):
        notebook_env.get_secret("HF_TOKEN")


def test_empty_secret_is_rejected(monkeypatch):
    """An empty token must fail here, not an hour later at checkpoint push."""
    _install_kaggle(monkeypatch, secret="")
    with pytest.raises(RuntimeError, match="empty value"):
        notebook_env.get_secret("HF_TOKEN")


def test_neither_platform_falls_back_to_environ(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-environ")
    assert notebook_env.get_secret("HF_TOKEN") == "from-environ"


def test_neither_platform_and_no_env_var_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="neither a Colab nor a Kaggle"):
        notebook_env.get_secret("HF_TOKEN")


# --------------------------------------------------------------------------
# paths and file delivery
# --------------------------------------------------------------------------


def test_working_root_per_platform(monkeypatch):
    _install_colab(monkeypatch)
    assert notebook_env.working_root() == Path("/content")

    monkeypatch.delitem(sys.modules, "google.colab")
    monkeypatch.delitem(sys.modules, "google")
    _install_kaggle(monkeypatch)
    assert notebook_env.working_root() == Path("/kaggle/working")


def test_repo_dir_lands_inside_kaggle_working(monkeypatch):
    """The clone must sit in /kaggle/working, the only roomy, persisted place."""
    _install_kaggle(monkeypatch)
    assert notebook_env.repo_dir() == Path("/kaggle/working/repo")


def test_working_root_off_platform_is_cwd(monkeypatch):
    assert notebook_env.working_root() == Path.cwd()


def test_deliver_file_downloads_on_colab(monkeypatch):
    downloads: list[str] = []
    _install_colab(monkeypatch, downloads=downloads)
    message = notebook_env.deliver_file("/content/metrics.zip")
    # Compared as paths, not strings: this test suite also runs on Windows, where
    # Path stringifies with backslashes. Colab itself is always Linux.
    assert [Path(p) for p in downloads] == [Path("/content/metrics.zip")]
    assert "metrics.zip" in message


def test_deliver_file_points_at_output_tab_on_kaggle(monkeypatch):
    """Kaggle has no download call; the message must say where to actually look."""
    _install_kaggle(monkeypatch)
    message = notebook_env.deliver_file("/kaggle/working/metrics.zip")
    assert "Output" in message


def test_deliver_file_survives_a_failed_colab_download(monkeypatch):
    """A failed download must not abort the notebook after a successful run."""

    def boom(_):
        raise OSError("no browser")

    _install_colab(monkeypatch)
    sys.modules["google.colab.files"].download = boom
    sys.modules["google.colab"].files.download = boom
    message = notebook_env.deliver_file("/content/metrics.zip")
    assert "Files pane" in message
