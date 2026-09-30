from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import toml
import yaml
from coverage import Coverage

from scripts import crap
from scripts.crap import (
    SOURCE_DIRECTORIES,
    Score,
    _functions,
    collect,
    main,
    score_file,
)


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


def test_coverage_collects_every_crap_source_tree() -> None:
    configuration = toml.loads(
        (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    pytest_config = configuration["tool"]["pytest"]["ini_options"]
    coverage_config = configuration["tool"]["coverage"]

    assert coverage_config["run"]["source"] == ["geoparser", "scripts", "tests"]
    assert {"--cov=geoparser", "--cov=scripts", "--cov=tests"} <= set(
        pytest_config["addopts"]
    )
    assert set(coverage_config["paths"]) == {"source", "scripts", "tests"}


def test_package_coverage_threshold_does_not_hide_crap_sources() -> None:
    workflow = Path(__file__).resolve().parents[3] / ".github/workflows/test.yml"
    configuration = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    coverage_steps = configuration["jobs"]["coverage"]["steps"]

    assert any(
        "coverage report --include='geoparser/*' --fail-under=" in step.get("run", "")
        for step in coverage_steps
    )


def test_crap_scope_includes_package_scripts_and_tests() -> None:
    assert SOURCE_DIRECTORIES == ("geoparser", "scripts", "tests")


def test_nested_functions_get_their_own_complexity_and_statement_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested.py"
    path.write_text(
        "def outer():\n"
        "    def inner(value):\n"
        "        if value:\n"
        "            return 1\n"
        "        return 0\n"
        "    return None\n",
        encoding="utf-8",
    )
    data_file = tmp_path / ".coverage"
    coverage = Coverage(data_file=str(data_file), config_file=False)
    coverage.start()
    namespace: dict[str, object] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    outer = cast(Callable[[], object], namespace["outer"])
    outer()
    coverage.stop()
    coverage.save()
    coverage.load()

    functions = _functions(path.read_text(encoding="utf-8"))
    scores = {score.name: score for score in score_file(coverage, path, tmp_path)}

    assert (
        [(function.name, function.complexity) for function in functions],
        scores["outer"].coverage,
        scores["outer.inner"].coverage,
        scores["outer.inner"].crap,
    ) == ([("outer", 1), ("outer.inner", 2)], 1.0, 0.0, 6.0)


def test_unmeasured_function_is_scored_as_uncovered(tmp_path: Path) -> None:
    path = tmp_path / "unmeasured.py"
    path.write_text(
        "def missing(value):\n    if value:\n        return 1\n    return 0\n",
        encoding="utf-8",
    )
    coverage = Coverage(data_file=str(tmp_path / "empty.coverage"), config_file=False)
    coverage.load()

    [score] = score_file(coverage, path, tmp_path)

    assert score.name == "missing"
    assert score.coverage == 0.0
    assert score.crap == 6.0


def test_collect_scores_functions_across_every_source_tree(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    for directory in SOURCE_DIRECTORIES:
        source_root = root / directory
        source_root.mkdir(parents=True)
        (source_root / "example.py").write_text(
            "def measured():\n    return 1\n", encoding="utf-8"
        )

    class CoverageData:
        def __init__(self, data_file: str) -> None:
            self.data_file = data_file

        def load(self) -> None:
            return None

        def analysis2(self, filename: str):
            missing = [2] if "/tests/" in filename else []
            return filename, [2], [], missing, ""

    monkeypatch.setattr(crap, "Coverage", CoverageData)

    scores = collect(root, tmp_path / ".coverage")

    assert [score.path for score in scores] == [
        "tests/example.py",
        "geoparser/example.py",
        "scripts/example.py",
    ]
    assert [score.coverage for score in scores] == [0.0, 1.0, 1.0]


def test_collect_fails_when_a_crap_source_tree_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    for directory in SOURCE_DIRECTORIES[:-1]:
        (root / directory).mkdir(parents=True)

    class CoverageData:
        def __init__(self, data_file: str) -> None:
            self.data_file = data_file

        def load(self) -> None:
            return None

    monkeypatch.setattr(crap, "Coverage", CoverageData)

    with pytest.raises(FileNotFoundError, match="CRAP source directory is missing"):
        collect(root, tmp_path / ".coverage")
