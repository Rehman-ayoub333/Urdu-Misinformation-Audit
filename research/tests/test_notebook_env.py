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
def _clean_env(monkeypatch):
    """Neither platform present, and no ambient Kaggle env var, unless a test says so."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "kaggle_secrets", raising=False)
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


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


def test_colab_wins_when_both_present(monkeypatch):
    """Not a real environment, but detection must still be deterministic."""
    _install_colab(monkeypatch, secret="from-colab")
    _install_kaggle(monkeypatch, secret="from-kaggle")
    assert notebook_env.get_secret("HF_TOKEN") == "from-colab"


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
