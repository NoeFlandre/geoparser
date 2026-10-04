import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import git_provenance
from scripts.benchmark import provenance
from scripts.panx_benchmark import runner


@pytest.mark.parametrize(
    ("short", "cwd", "revision_arguments"),
    [
        (True, Path("/repository"), ["git", "rev-parse", "--short", "HEAD"]),
        (False, None, ["git", "rev-parse", "HEAD"]),
    ],
)
@pytest.mark.parametrize(
    ("commit_stdout", "status_stdout", "expected"),
    [
        ("abc123\n", "\n", "abc123"),
        ("\n abc123 \n", " \t \n", "abc123"),
        ("abc123\n", " M file.py \n", "abc123-dirty"),
    ],
)
def test_commit_id_probes_commit_and_porcelain_status(
    monkeypatch,
    short,
    cwd,
    revision_arguments,
    commit_stdout,
    status_stdout,
    expected,
):
    calls = []
    outputs = iter((commit_stdout, status_stdout))

    def run(arguments, **options):
        calls.append((arguments, options))
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(git_provenance.subprocess, "run", run)

    result = git_provenance.commit_id(short=short, cwd=cwd)

    assert result == expected
    assert calls == [
        (
            revision_arguments,
            {"cwd": cwd, "capture_output": True, "text": True, "check": True},
        ),
        (
            ["git", "status", "--porcelain"],
            {"cwd": cwd, "capture_output": True, "text": True, "check": True},
        ),
    ]


@pytest.mark.parametrize("short,cwd", [(True, Path("/repository")), (False, None)])
@pytest.mark.parametrize("failed_command", [0, 1])
@pytest.mark.parametrize(
    "error",
    [OSError("git missing"), subprocess.CalledProcessError(1, "git")],
)
def test_commit_id_returns_unknown_when_either_probe_fails(
    monkeypatch, short, cwd, failed_command, error
):
    calls = []

    def run(*_arguments, **_options):
        command_index = len(calls)
        calls.append(command_index)
        if command_index == failed_command:
            raise error
        return SimpleNamespace(stdout="abc123\n")

    monkeypatch.setattr(git_provenance.subprocess, "run", run)

    assert git_provenance.commit_id(short=short, cwd=cwd) == "unknown"
    assert len(calls) == failed_command + 1


def test_source_commit_uses_short_hash_and_repository_root(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        provenance.git_provenance,
        "commit_id",
        lambda **options: calls.append(options) or "abc123",
    )

    assert provenance.source_commit(tmp_path) == "abc123"
    assert calls == [{"short": True, "cwd": tmp_path}]


def test_panx_commit_id_uses_full_hash_and_inherited_cwd(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner.git_provenance,
        "commit_id",
        lambda **options: calls.append(options) or "abc123",
    )

    assert runner._commit_id() == "abc123"
    assert calls == [{"short": False, "cwd": None}]
