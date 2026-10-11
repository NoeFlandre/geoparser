"""The reviewed ranker environment: its requirements, its version check and its JSON contract."""

import importlib.metadata
import io
import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.unitoprank_benchmark import isolated_ranker
from scripts.unitoprank_benchmark.isolated_ranker import (
    REPOSITORY_ROOT,
    REQUIREMENTS,
    installed_mismatches,
    rank_in_reviewed_environment,
    reviewed_versions,
)
from scripts.unitoprank_benchmark.pins import REVIEWED_BLOBS, git_blob_id

REVIEWED = {"numpy": "2.4.2", "rapidfuzz": "3.14.3", "requests": "2.32.5"}


def test_the_requirements_file_is_byte_identical_to_the_reviewed_upstream_file():
    assert git_blob_id(REQUIREMENTS.read_bytes()) == REVIEWED_BLOBS["requirements.txt"]


def test_the_requirements_file_keeps_its_bytes_in_an_autocrlf_checkout(tmp_path: Path):
    """A Windows checkout with autocrlf must still hold the reviewed LF bytes."""
    relative = REQUIREMENTS.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    source = tmp_path / "source"
    for name in (relative, ".gitattributes"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPOSITORY_ROOT / name).read_bytes())
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    commands = (
        ("init",),
        ("-c", "core.autocrlf=false", "add", "."),
        (
            "-c",
            "core.autocrlf=true",
            "checkout-index",
            "--all",
            f"--prefix={checkout.as_posix()}/",
        ),
    )
    for command in commands:
        subprocess.run(["git", *command], cwd=source, capture_output=True, check=True)
    assert (
        git_blob_id((checkout / relative).read_bytes())
        == (REVIEWED_BLOBS["requirements.txt"])
    )


def test_the_requirements_file_pins_the_three_reviewed_packages():
    assert reviewed_versions(REQUIREMENTS.read_text()) == REVIEWED


def test_a_requirement_without_an_exact_pin_is_refused():
    with pytest.raises(ValueError, match="not an exact pin: 'numpy>=2'"):
        reviewed_versions("numpy>=2\n")


def test_an_installed_version_that_differs_from_its_pin_is_named():
    installed = importlib.metadata.version("rapidfuzz")
    assert installed_mismatches({"rapidfuzz": installed}) == []
    assert installed_mismatches({"rapidfuzz": "0.0.1"}) == [
        f"rapidfuzz is {installed}, reviewed 0.0.1"
    ]


def test_main_prints_the_ranking_for_the_request_on_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path
):
    seen: dict[str, object] = {}

    class Config:
        def __init__(self, **fields: object) -> None:
            self.fields = fields

    def rank_toponyms(*, text, toponyms, candidates_by_toponym, config):
        seen.update(
            text=text,
            toponyms=toponyms,
            candidates=candidates_by_toponym,
            config=config.fields,
        )
        # The configuration echo can hold a set, which JSON cannot encode.
        return {
            "text": text,
            "used_config": {"skip_toponyms": {"paris"}},
            "ranked_candidates_by_toponym": {"paris": [{"address": "Paris"}]},
        }

    monkeypatch.setattr(isolated_ranker, "installed_mismatches", lambda _pins: [])
    monkeypatch.setattr(
        isolated_ranker,
        "load_rank_toponyms",
        lambda checkout: (rank_toponyms, Config),
    )
    request = {
        "text": "Paris",
        "toponyms": [{"text": "Paris", "start": 0, "end": 5}],
        "candidates_by_toponym": {"paris": []},
        "config": {"top_n": 3},
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))

    assert isolated_ranker.main([str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "ranked_candidates_by_toponym": {"paris": [{"address": "Paris"}]}
    }
    assert seen == {
        "text": "Paris",
        "toponyms": request["toponyms"],
        "candidates": {"paris": []},
        "config": {"top_n": 3},
    }


def test_main_refuses_to_import_the_checkout_when_a_version_differs(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path
):
    monkeypatch.setattr(
        isolated_ranker,
        "installed_mismatches",
        lambda _pins: ["rapidfuzz is 3.14.6, reviewed 3.14.3"],
    )

    def must_not_load(_checkout):
        message = "the checkout was imported"
        raise AssertionError(message)

    monkeypatch.setattr(isolated_ranker, "load_rank_toponyms", must_not_load)
    assert isolated_ranker.main([str(tmp_path)]) == 2
    assert capsys.readouterr().err.strip() == "rapidfuzz is 3.14.6, reviewed 3.14.3"


def test_a_run_that_fails_in_its_interpreter_raises_with_the_reason(tmp_path: Path):
    with pytest.raises(RuntimeError, match="the reviewed ranker did not run"):
        rank_in_reviewed_environment(
            tmp_path / "missing",
            sys.executable,
            text="",
            toponyms=[],
            candidates_by_toponym={},
            config={},
        )


def test_a_relative_interpreter_path_is_read_from_the_callers_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The child runs from the repository root, so a relative path must be made absolute first."""
    base = tmp_path.resolve()
    try:
        relative = os.path.relpath(Path(sys.executable).absolute(), base)
    except ValueError:  # the interpreter is on another drive
        pytest.skip("no relative path to the interpreter on this platform")
    monkeypatch.chdir(base)
    assert not Path(relative).is_absolute()
    with pytest.raises(RuntimeError, match="the reviewed ranker did not run"):
        rank_in_reviewed_environment(
            base / "checkout",
            relative,
            text="",
            toponyms=[],
            candidates_by_toponym={},
            config={},
        )


def test_an_interpreter_path_is_made_absolute_and_a_bare_name_is_kept():
    assert Path(isolated_ranker._interpreter_command(".venv/bin/python")).is_absolute()
    assert Path(isolated_ranker._interpreter_command("./python")).is_absolute()
    assert isolated_ranker._interpreter_command("python3") == "python3"


def test_the_child_guard_allows_loopback_and_refuses_external_addresses():
    with socket.create_server(("127.0.0.1", 0)) as listener:
        port = listener.getsockname()[1]
        with isolated_ranker.external_network_refused():
            socket.create_connection(("127.0.0.1", port), timeout=5).close()
            with pytest.raises(
                RuntimeError, match="network access is disabled for the reviewed ranker"
            ):
                socket.create_connection(("192.0.2.1", 9), timeout=1)


def test_the_child_guard_judges_each_kind_of_address():
    assert isolated_ranker._is_loopback_host(b"LOCALHOST.")
    assert isolated_ranker._is_loopback_host("127.0.0.1")
    assert not isolated_ranker._is_loopback_host("example.invalid")
    isolated_ranker._require_loopback("/run/unix-socket")
    isolated_ranker._require_loopback(("localhost", 80))
    for address in (("192.0.2.1", 80), 42):
        with pytest.raises(
            RuntimeError, match="network access is disabled for the reviewed ranker"
        ):
            isolated_ranker._require_loopback(address)


def test_the_child_guard_removes_its_patches_on_exit():
    before = (socket.getaddrinfo, socket.socket.connect, socket.socket.sendto)
    with isolated_ranker.external_network_refused():
        assert socket.socket.connect is not before[1]
    after = (socket.getaddrinfo, socket.socket.connect, socket.socket.sendto)
    assert all(old is new for old, new in zip(before, after, strict=True))


def test_a_child_that_attempts_a_connection_while_ranking_fails(tmp_path: Path):
    """The pytest guard does not reach the child, so the child refuses the connection itself."""
    code = textwrap.dedent(
        """
        import socket
        import sys

        sys.path.insert(0, sys.argv[1])
        from scripts.unitoprank_benchmark import isolated_ranker

        isolated_ranker.installed_mismatches = lambda _pins: []

        def load(_checkout):
            socket.create_connection(("192.0.2.1", 9), timeout=1)
            raise AssertionError("the connection was not refused")

        isolated_ranker.load_rank_toponyms = load
        sys.exit(isolated_ranker.main([sys.argv[2]]))
        """
    )
    request = {
        "text": "",
        "toponyms": [],
        "candidates_by_toponym": {},
        "config": {},
    }
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(REPOSITORY_ROOT), str(tmp_path)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "network access is disabled for the reviewed ranker" in completed.stderr
    assert "AssertionError" not in completed.stderr


def test_a_shadow_module_on_pythonpath_is_not_imported_by_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """PYTHONPATH must not reach the child: sitecustomize runs before any version check."""
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    marker = tmp_path / "shadow-ran"
    code = f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
    (shadow / "sitecustomize.py").write_text(code)
    (shadow / "rapidfuzz.py").write_text(code)
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    # The child refuses to run here, so the call raises; what matters is that
    # nothing from the shadow directory executed on the way.
    with pytest.raises(RuntimeError, match="the reviewed ranker did not run"):
        rank_in_reviewed_environment(
            tmp_path / "checkout",
            sys.executable,
            text="",
            toponyms=[],
            candidates_by_toponym={},
            config={},
        )
    assert not marker.exists()
