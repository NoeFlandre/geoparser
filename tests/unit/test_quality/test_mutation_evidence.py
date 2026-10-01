"""Contracts for the per-mutant CI evidence artifact."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import mutation_evidence


class _MutmutRunner:
    """Small subprocess double for result listing and mutant display calls."""

    def __init__(self, result_output: str) -> None:
        self.result_output = result_output
        self.commands = []

    def __call__(self, command, **_kwargs):
        self.commands.append(tuple(command))
        if command[0] == "git":
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[-3:] == ("results", "--all", "true"):
            return SimpleNamespace(
                returncode=0,
                stdout=self.result_output,
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=f"diff for {command[-1]}",
            stderr="",
        )


def test_parse_results_keeps_mutant_ids_outcomes_and_unparsed_lines() -> None:
    output = (
        "\x1b[32mgeoparser.services.recognition.fit__mutmut_3: survived\x1b[0m\n"
        "geoparser.services.recognition.fit__mutmut_8: timeout\n"
        "Geoparser summary: 2 results\n"
    )

    records, unparsed = mutation_evidence.parse_results(output)

    assert records == [
        {"id": "geoparser.services.recognition.fit__mutmut_3", "outcome": "survived"},
        {"id": "geoparser.services.recognition.fit__mutmut_8", "outcome": "timeout"},
    ]
    assert unparsed == ["Geoparser summary: 2 results"]


def test_select_results_applies_changed_module_patterns() -> None:
    records = [
        {"id": "geoparser.services.recognition.fit__mutmut_1", "outcome": "timeout"},
        {"id": "geoparser.services.resolution.fit__mutmut_2", "outcome": "killed"},
    ]

    selected = mutation_evidence.select_results(
        records, ["geoparser.services.recognition.*"]
    )

    assert selected == records[:1]
    assert mutation_evidence.select_results(records, []) == records


def _sample_evidence(monkeypatch, tmp_path: Path):
    result_lines = "\n".join(
        (
            "geoparser.services.recognition.fit__mutmut_3: survived",
            "geoparser.services.recognition.fit__mutmut_8: timeout",
            "geoparser.services.recognition.fit__mutmut_9: killed",
            "geoparser.services.resolution.fit__mutmut_2: survived",
        )
    )
    runner = _MutmutRunner(result_lines)
    monkeypatch.setattr(mutation_evidence.subprocess, "run", runner)
    stats_path = tmp_path / "mutmut-cicd-stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "killed": 1,
                "survived": 1,
                "timeout": 1,
                "no_tests": 0,
                "total": 3,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "evidence"

    result = mutation_evidence.write_evidence(
        output_dir,
        stats_path,
        ["geoparser.services.recognition.*"],
        0,
        0,
        1,
    )

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    return result, report, runner, output_dir, result_lines


def test_evidence_records_run_status(monkeypatch, tmp_path: Path):
    result, report, _runner, _output_dir, _result_lines = _sample_evidence(
        monkeypatch, tmp_path
    )

    assert result == 0
    assert report["exit_codes"] == {
        "mutation_run": 0,
        "stats_export": 0,
        "mutation_gate": 1,
        "results_capture": 0,
    }


def test_evidence_records_outcome_counts(monkeypatch, tmp_path: Path):
    _result, report, _runner, _output_dir, _result_lines = _sample_evidence(
        monkeypatch, tmp_path
    )

    assert report["unaccounted_mutants"] == 0
    assert report["counts_by_outcome"] == {"survived": 1, "timeout": 1, "killed": 1}
    assert report["mutant_count"] == 3
    assert report["metadata"]["selection_mode"] == "changed_modules"


def test_evidence_records_unresolved_diagnostics(monkeypatch, tmp_path: Path):
    _result, report, _runner, _output_dir, _result_lines = _sample_evidence(
        monkeypatch, tmp_path
    )

    assert report["mutants"][0]["diagnostic"]["stdout"].endswith("mutmut_3")
    assert report["mutants"][1]["diagnostic"]["stdout"].endswith("mutmut_8")
    assert report["mutants"][2]["diagnostic"] is None


def test_evidence_records_replay_command_and_diagnostic_count(
    monkeypatch, tmp_path: Path
):
    _result, report, runner, _output_dir, _result_lines = _sample_evidence(
        monkeypatch, tmp_path
    )

    assert report["mutants"][0]["replay_command"].endswith(
        "geoparser.services.recognition.fit__mutmut_3"
    )
    assert sum(command[-2] == "show" for command in runner.commands) == 2


def test_evidence_persists_raw_results_and_exported_stats(monkeypatch, tmp_path: Path):
    _result, report, _runner, output_dir, result_lines = _sample_evidence(
        monkeypatch, tmp_path
    )

    assert (output_dir / "mutmut-results.txt").read_text(encoding="utf-8") == (
        result_lines
    )
    stats_file = output_dir / "mutmut-cicd-stats.json"
    assert json.loads(stats_file.read_text(encoding="utf-8")) == report["stats"]


def _unchecked_evidence(monkeypatch, tmp_path: Path):
    result_lines = "geoparser.services.recognition.fit__mutmut_3: not checked"
    runner = _MutmutRunner(result_lines)
    monkeypatch.setattr(mutation_evidence.subprocess, "run", runner)
    output_dir = tmp_path / "evidence"

    mutation_evidence.write_evidence(
        output_dir,
        tmp_path / "missing-stats.json",
        [],
        1,
        0,
        1,
    )

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    return report, runner


def test_unchecked_evidence_records_missing_stats(monkeypatch, tmp_path: Path):
    report, _runner = _unchecked_evidence(monkeypatch, tmp_path)

    assert "FileNotFoundError" in report["stats_error"]
    assert report["stats"] is None
    assert report["unaccounted_mutants"] is None
    assert report["mutants"][0]["diagnostic"] is None


def test_unchecked_evidence_skips_redundant_mutant_diagnostics(
    monkeypatch, tmp_path: Path
):
    _report, runner = _unchecked_evidence(monkeypatch, tmp_path)

    assert sum(command[-2] == "show" for command in runner.commands) == 0


def test_main_parses_cli_and_writes_evidence(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mutation_evidence.subprocess, "run", _MutmutRunner(""))
    output_dir = tmp_path / "artifact"
    stats_path = tmp_path / "missing.json"
    result = mutation_evidence.main(
        [
            "--evidence-dir",
            str(output_dir),
            "--stats",
            str(stats_path),
            "--run-exit-code",
            "0",
            "--export-exit-code",
            "1",
            "--gate-exit-code",
            "1",
        ]
    )

    assert result == 0
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["metadata"]["selection_mode"] == "full"
