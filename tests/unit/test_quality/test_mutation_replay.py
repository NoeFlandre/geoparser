"""Contracts for the bounded, CI-only exact-mutant replay path."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from scripts import mutation_replay
from tests.conftest import PROJECT_ROOT

ALLOWLIST = PROJECT_ROOT / "scripts" / "mutation_replay_allowlist.json"
CURRENT_SHA = "f0f07a630fda3be4b758d27690bd12a2c280897c"


def _known_mutant_ids() -> list[str]:
    return json.loads(ALLOWLIST.read_text(encoding="utf-8"))["mutant_ids"]


def _stub_mutmut_runner(
    mutant_id: str,
    outcome: str = "timeout",
    run_code: int = 0,
    results_code: int = 0,
    commands: list[list[str]] | None = None,
):
    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        is_results = command[3] == "results"
        if commands is not None:
            commands.append(command)
        stdout = f"{mutant_id}: {outcome}\n" if is_results else "replayed\n"
        return CompletedProcess(
            command,
            results_code if is_results else run_code,
            stdout=stdout,
            stderr="",
        )

    return run


def _run_timeout_replay(
    tmp_path: Path, commands: list[list[str]]
) -> tuple[int, dict[str, Any]]:
    mutant_id = _known_mutant_ids()[0]
    result = mutation_replay.execute_replay(
        mutant_id,
        ALLOWLIST,
        CURRENT_SHA,
        CURRENT_SHA,
        tmp_path / "evidence",
        _stub_mutmut_runner(mutant_id, commands=commands),
    )
    report = json.loads(
        (tmp_path / "evidence" / "replay-report.json").read_text(encoding="utf-8")
    )
    return result, report


def test_allowlist_records_the_complete_versioned_timeout_inventory() -> None:
    allowlist = json.loads(ALLOWLIST.read_text(encoding="utf-8"))

    assert allowlist["source"] == {
        "workflow_run_id": "36842110918",
        "pull_request_head_sha": "f0f07a630fda3be4b758d27690bd12a2c280897c",
        "checkout_sha": "da46b4841404493012e07f2e8df6165ffe8f02ec",
        "report_sha256": "03431720966578a31913849bf3d1b945cc7a3294d3d9312f95691c8c6c9f1c0a",
        "timeout_count": 27,
        "replay_source_sha": CURRENT_SHA,
        "source_tree_sha": "fb77bc0a9cb2516c16343c6c39d971560f6a9878",
    }
    assert len(allowlist["mutant_ids"]) == 27
    assert allowlist["max_selected_per_dispatch"] == 8


def test_selection_accepts_only_an_exact_allowed_id_for_the_selected_sha() -> None:
    mutant_id = _known_mutant_ids()[0]

    selected, allowlist = mutation_replay.validate_selection(
        f"{mutant_id}\n", ALLOWLIST, CURRENT_SHA, CURRENT_SHA
    )

    assert selected == [mutant_id]
    assert allowlist["source"]["timeout_count"] == 27


def test_selection_rejects_globs_and_other_unlisted_ids() -> None:
    with pytest.raises(mutation_replay.ReplaySelectionError, match="outside"):
        mutation_replay.validate_selection(
            "geoparser.modules.resolvers.*\n", ALLOWLIST, CURRENT_SHA, CURRENT_SHA
        )


def test_selection_rejects_duplicates_and_dispatches_over_eight_ids() -> None:
    ids = _known_mutant_ids()

    with pytest.raises(mutation_replay.ReplaySelectionError, match="duplicate"):
        mutation_replay.validate_selection(
            f"{ids[0]}\n{ids[0]}\n", ALLOWLIST, CURRENT_SHA, CURRENT_SHA
        )
    with pytest.raises(mutation_replay.ReplaySelectionError, match="at most 8"):
        mutation_replay.validate_selection(
            "\n".join(ids[:9]), ALLOWLIST, CURRENT_SHA, CURRENT_SHA
        )


def test_selection_rejects_a_mismatched_or_abbreviated_sha() -> None:
    mutant_id = _known_mutant_ids()[0]

    with pytest.raises(mutation_replay.ReplaySelectionError, match="does not match"):
        mutation_replay.validate_selection(mutant_id, ALLOWLIST, CURRENT_SHA, "0" * 40)
    with pytest.raises(mutation_replay.ReplaySelectionError, match="full lowercase"):
        mutation_replay.validate_selection(mutant_id, ALLOWLIST, "fb896a2", CURRENT_SHA)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda allowlist: allowlist.update(schema_version=2), "schema"),
        (lambda allowlist: allowlist.update(mutant_ids=[]), "no mutant IDs"),
        (
            lambda allowlist: allowlist.update(
                mutant_ids=[allowlist["mutant_ids"][0]] * 27
            ),
            "duplicate",
        ),
        (
            lambda allowlist: allowlist.update(mutant_ids=[None] * 31),
            "must be strings",
        ),
        (
            lambda allowlist: allowlist["mutant_ids"].__setitem__(
                0, "geoparser.*__mutmut_1"
            ),
            "invalid literal",
        ),
        (
            lambda allowlist: allowlist.update(max_selected_per_dispatch=9),
            "cap",
        ),
    ],
)
def test_allowlist_rejects_malformed_inventory(
    tmp_path: Path, edit, message: str
) -> None:
    allowlist = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    edit(allowlist)
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps(allowlist), encoding="utf-8")

    with pytest.raises(mutation_replay.ReplaySelectionError, match=message):
        mutation_replay._load_allowlist(path)


@pytest.mark.parametrize("selection", ["\n", " id\n", "id \n"])
def test_selection_rejects_blank_or_padded_lines(selection: str) -> None:
    with pytest.raises(mutation_replay.ReplaySelectionError):
        mutation_replay._selection_lines(selection, maximum=8)


def test_cli_rejects_an_empty_selection_before_starting_a_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    selection = tmp_path / "ids.txt"
    selection.write_text("\n", encoding="utf-8")

    result = mutation_replay.main(
        [
            "--selection-file",
            str(selection),
            "--expected-sha",
            CURRENT_SHA,
            "--actual-sha",
            CURRENT_SHA,
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ]
    )

    assert result == 2
    assert "Select at least one exact mutant ID" in capsys.readouterr().err


def test_cli_validation_only_outputs_the_checked_literal_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mutant_id = _known_mutant_ids()[0]
    selection = tmp_path / "ids.txt"
    selection.write_text(f"{mutant_id}\n", encoding="utf-8")

    result = mutation_replay.main(
        [
            "--selection-file",
            str(selection),
            "--expected-sha",
            CURRENT_SHA,
            "--actual-sha",
            CURRENT_SHA,
            "--validate-only",
        ]
    )

    assert result == 0
    assert capsys.readouterr().out == f"{mutant_id}\n"


@pytest.mark.parametrize(("run_code", "results_code"), [(1, 0), (0, 1)])
def test_replay_fails_when_mutmut_command_fails(
    tmp_path: Path, run_code: int, results_code: int
) -> None:
    mutant_id = _known_mutant_ids()[0]
    runner = _stub_mutmut_runner(
        mutant_id, run_code=run_code, results_code=results_code
    )

    result = mutation_replay.execute_replay(
        mutant_id,
        ALLOWLIST,
        CURRENT_SHA,
        CURRENT_SHA,
        tmp_path / "evidence",
        runner,
    )

    assert result == 1


def test_replay_records_timeout_as_inconclusive_without_counting_a_kill(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    result, report = _run_timeout_replay(tmp_path, commands)

    assert result == 0
    assert report["counts_by_outcome"] == {"timeout": 1}
    assert report["timeouts_are_inconclusive"] is True
    assert "1 timeout (inconclusive)" in report["diagnostic_summary"]


def test_replay_passes_only_allowlisted_ids_to_serial_mutmut(
    tmp_path: Path,
) -> None:
    mutant_id = _known_mutant_ids()[0]
    commands: list[list[str]] = []
    _result, _report = _run_timeout_replay(tmp_path, commands)

    assert commands[0][3:6] == ["run", "--max-children", "1"]
    assert commands[0][-1] == mutant_id
    assert "timeout" not in commands[0]


def test_replay_fails_on_a_survivor(tmp_path: Path) -> None:
    mutant_id = _known_mutant_ids()[0]
    runner = _stub_mutmut_runner(mutant_id, outcome="survived")

    result = mutation_replay.execute_replay(
        mutant_id,
        ALLOWLIST,
        CURRENT_SHA,
        CURRENT_SHA,
        tmp_path / "evidence",
        runner,
    )

    assert result == 1


def test_selection_rejects_stale_ids_even_when_requested_sha_matches_checkout():
    with pytest.raises(mutation_replay.ReplaySelectionError, match="revision-specific"):
        mutation_replay.validate_selection(
            _known_mutant_ids()[0], ALLOWLIST, "a" * 40, "a" * 40
        )
