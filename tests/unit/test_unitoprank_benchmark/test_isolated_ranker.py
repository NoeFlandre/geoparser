"""The reviewed ranker environment: its requirements, its version check and its JSON contract."""

import importlib.metadata
import io
import json
import subprocess
import sys
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
