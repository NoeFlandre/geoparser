import hashlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.benchmark import provenance


def test_gazetteer_artifact_sha256_hashes_the_installed_file(monkeypatch, tmp_path):
    artifact = tmp_path / "geonames.db"
    artifact.write_bytes(b"known gazetteer bytes")
    monkeypatch.setattr(
        "geoparser.gazetteer.artifact.artifact_path", lambda name: artifact
    )

    actual = provenance.gazetteer_artifact_sha256("geonames")

    assert actual == hashlib.sha256(b"known gazetteer bytes").hexdigest()


def test_gazetteer_artifact_sha256_fails_when_artifact_is_missing(
    monkeypatch, tmp_path
):
    artifact = tmp_path / "missing.db"
    monkeypatch.setattr(
        "geoparser.gazetteer.artifact.artifact_path", lambda name: artifact
    )

    with pytest.raises(FileNotFoundError, match="Gazetteer artifact not found"):
        provenance.gazetteer_artifact_sha256("geonames")


@pytest.mark.parametrize(
    ("status", "expected"), [("", "abc123"), (" M file.py", "abc123-dirty")]
)
def test_source_commit_marks_dirty_trees(monkeypatch, tmp_path, status, expected):
    outputs = iter(["abc123\n", f"{status}\n"])
    monkeypatch.setattr(
        provenance.git_provenance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=next(outputs)),
    )

    assert provenance.source_commit(tmp_path) == expected


@pytest.mark.parametrize(
    "error", [OSError("git missing"), subprocess.CalledProcessError(1, "git")]
)
def test_source_commit_returns_unknown_when_git_cannot_report_state(
    monkeypatch, tmp_path, error
):
    monkeypatch.setattr(
        provenance.git_provenance.subprocess, "run", Mock(side_effect=error)
    )

    assert provenance.source_commit(tmp_path) == "unknown"


def _install_test_torch(monkeypatch, *, cuda: bool) -> None:
    """Provide deterministic Torch hardware facts to the environment probe."""
    fake_torch = SimpleNamespace(
        __version__="test-version",
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            get_device_name=lambda index: "test GPU",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_environment_records_cpu_without_a_scheduler_job(monkeypatch):
    _install_test_torch(monkeypatch, cuda=False)

    facts = provenance.environment()

    assert {
        key: facts.get(key) for key in ("torch", "cuda_available", "job_id", "gpu")
    } == {
        "torch": "test-version",
        "cuda_available": False,
        "job_id": None,
        "gpu": None,
    }
    assert facts["started_at"].endswith("+00:00")


def test_environment_records_scheduler_and_gpu(monkeypatch):
    _install_test_torch(monkeypatch, cuda=True)

    facts = provenance.environment("job-17")

    assert {
        key: facts.get(key) for key in ("torch", "cuda_available", "job_id", "gpu")
    } == {
        "torch": "test-version",
        "cuda_available": True,
        "job_id": "job-17",
        "gpu": "test GPU",
    }
    assert facts["started_at"].endswith("+00:00")
