"""Tests for the M4-6 results-durability push.

The contract worth protecting here is unusual and easy to erode: pushing results
must NEVER raise, because a failed upload must not destroy a finished training
run — but it must also never quietly claim success, because reporting success it
did not achieve is precisely how Milestone 4's metrics were lost in the first
place. Both halves are asserted below.

huggingface_hub is stubbed throughout; nothing here touches the network.
"""

from __future__ import annotations

import sys
import types

import pytest

from research.src.evaluation import results_push


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("HF_STAGING_PREFIX", "test-user")


def _install_hub(monkeypatch, *, upload_raises=None, create_raises=None, listed=()):
    """Stub huggingface_hub.HfApi, recording what it was asked to do."""
    calls: dict[str, list] = {"uploaded": [], "created": []}

    class HfApi:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
            if create_raises:
                raise create_raises
            calls["created"].append((repo_id, repo_type))

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None):
            if upload_raises:
                raise upload_raises
            calls["uploaded"].append(path_in_repo)

        def list_repo_files(self, repo_id, repo_type=None):
            return list(listed)

    module = types.ModuleType("huggingface_hub")
    module.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def _metrics_file(tmp_path, name="C_mbert_ax_to_grind_test_seed42.json"):
    path = tmp_path / name
    path.write_text('{"experiment_id": "C"}', encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_uploads_each_file_individually(monkeypatch, tmp_path):
    """Raw JSONs, not just a zip — each must be separately fetchable."""
    calls = _install_hub(monkeypatch)
    files = [_metrics_file(tmp_path, f"C_m_ax_to_grind_test_seed{s}.json") for s in (42, 123)]

    status = results_push.push_result_files(files)

    assert status["pushed"] is True
    assert status["failed"] == []
    assert calls["uploaded"] == [
        "milestone4/metrics/C_m_ax_to_grind_test_seed42.json",
        "milestone4/metrics/C_m_ax_to_grind_test_seed123.json",
    ]


def test_targets_a_dataset_repo_in_the_staging_namespace(monkeypatch, tmp_path):
    calls = _install_hub(monkeypatch)
    results_push.push_result_files([_metrics_file(tmp_path)])
    assert calls["created"] == [("test-user/urdu-misinfo-results-staging", "dataset")]


def test_empty_input_is_a_no_op_success(monkeypatch):
    _install_hub(monkeypatch)
    assert results_push.push_result_files([])["pushed"] is True


# --------------------------------------------------------------------------
# never raises, never lies
# --------------------------------------------------------------------------


def test_upload_failure_does_not_raise(monkeypatch, tmp_path):
    """A dead Hub must not kill a finished training run."""
    _install_hub(monkeypatch, upload_raises=OSError("503 Service Unavailable"))
    status = results_push.push_result_files([_metrics_file(tmp_path)], attempts=1)
    assert status["pushed"] is False
    assert status["failed"] == ["C_mbert_ax_to_grind_test_seed42.json"]


def test_upload_failure_is_reported_not_swallowed(monkeypatch, tmp_path, capsys):
    _install_hub(monkeypatch, upload_raises=OSError("503"))
    results_push.push_result_files([_metrics_file(tmp_path)], attempts=1)
    assert "FAILED" in capsys.readouterr().out


def test_upload_is_retried_before_giving_up(monkeypatch, tmp_path):
    """Hub 5xx responses are transient often enough to be worth a retry."""
    monkeypatch.setattr(results_push.time, "sleep", lambda _: None)
    attempts = {"n": 0}

    calls = _install_hub(monkeypatch)
    real_api = sys.modules["huggingface_hub"].HfApi

    class Flaky(real_api):
        def upload_file(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise OSError("503")
            calls["uploaded"].append(kwargs["path_in_repo"])

    sys.modules["huggingface_hub"].HfApi = Flaky

    status = results_push.push_result_files([_metrics_file(tmp_path)], attempts=3)
    assert attempts["n"] == 3
    assert status["pushed"] is True


def test_missing_credentials_does_not_raise_but_reports_failure(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _install_hub(monkeypatch)
    status = results_push.push_result_files([_metrics_file(tmp_path)])
    assert status["pushed"] is False
    assert "HF_TOKEN" in status["reason"]


def test_repo_creation_failure_does_not_raise(monkeypatch, tmp_path):
    _install_hub(monkeypatch, create_raises=OSError("403 forbidden"))
    status = results_push.push_result_files([_metrics_file(tmp_path)])
    assert status["pushed"] is False


def test_nonexistent_file_is_reported_as_failed(monkeypatch, tmp_path):
    """Never report success for a file that was never uploaded."""
    _install_hub(monkeypatch)
    status = results_push.push_result_files([tmp_path / "not-there.json"])
    assert status["pushed"] is False
    assert status["failed"] == ["not-there.json"]


def test_dry_run_uploads_nothing(monkeypatch, tmp_path):
    calls = _install_hub(monkeypatch)
    status = results_push.push_result_files([_metrics_file(tmp_path)], dry_run=True)
    assert status["pushed"] is False
    assert status["reason"] == "dry run"
    assert calls["uploaded"] == []
    assert calls["created"] == []


# --------------------------------------------------------------------------
# verification — the half that is allowed to fail a run
# --------------------------------------------------------------------------


def test_verify_passes_when_everything_is_present(monkeypatch):
    _install_hub(monkeypatch, listed=["milestone4/metrics/a.json", "milestone4/metrics/b.json"])
    result = results_push.verify_results_uploaded(["a.json", "b.json"])
    assert result["verified"] is True
    assert result["missing"] == []


def test_verify_names_what_is_missing(monkeypatch):
    _install_hub(monkeypatch, listed=["milestone4/metrics/a.json"])
    result = results_push.verify_results_uploaded(["a.json", "b.json"])
    assert result["verified"] is False
    assert result["missing"] == ["b.json"]


def test_verify_fails_closed_when_the_hub_is_unreachable(monkeypatch):
    """Unverifiable must never read as verified."""
    module = types.ModuleType("huggingface_hub")

    class HfApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type=None):
            raise OSError("network down")

    module.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    result = results_push.verify_results_uploaded(["a.json"])
    assert result["verified"] is False
    assert result["missing"] == ["a.json"]
