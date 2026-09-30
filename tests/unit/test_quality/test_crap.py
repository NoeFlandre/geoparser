from pathlib import Path

import yaml

from scripts.crap import Score, main


def _run_gate(score: Score, data_file: Path, monkeypatch) -> int:
    data_file.touch()
    monkeypatch.setattr("scripts.crap.collect", lambda *args: [score])
    return main(["--max-crap", "6", "--data-file", str(data_file)])


def test_crap_gate_rejects_a_score_equal_to_the_exclusive_limit(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    score = Score("example.py", "branchy", 1, complexity=6, coverage=1.0)

    assert _run_gate(score, tmp_path / ".coverage", monkeypatch) == 1
    assert "exceed the CRAP threshold of 6" in capsys.readouterr().err


def test_crap_gate_accepts_a_score_below_the_exclusive_limit(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    score = Score("example.py", "simple", 1, complexity=5, coverage=1.0)

    assert _run_gate(score, tmp_path / ".coverage", monkeypatch) == 0
    assert "within the CRAP threshold of 6" in capsys.readouterr().out


def test_ci_uses_the_same_exclusive_crap_limit_as_the_gate() -> None:
    workflow = Path(__file__).resolve().parents[3] / ".github/workflows/test.yml"
    configuration = yaml.safe_load(workflow.read_text(encoding="utf-8"))

    assert configuration["env"]["CRAP_THRESHOLD"] == "6"
