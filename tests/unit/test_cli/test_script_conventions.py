"""Characterise the command-line behaviour of the release and architecture scripts.

These tests pin observable behaviour (exit codes, stdout, stderr, help text and
both invocation styles) so that restructuring how each script builds its
argument parser cannot change what a caller sees.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

import pytest

from scripts import changelog, check_architecture
from tests.conftest import PROJECT_ROOT

CHANGELOG_SCRIPT = PROJECT_ROOT / "scripts" / "changelog.py"
ARCHITECTURE_SCRIPT = PROJECT_ROOT / "scripts" / "check_architecture.py"
CHANGELOG_TEXT = "# Changelog\n\n## [1.0.0]\n\nShipped.\n\n## [0.9.0]\n\n   \n"


def _argparse_scripts() -> tuple[Path, ...]:
    """Every script under scripts/ that builds an argument parser."""
    return tuple(
        sorted(
            path
            for path in (PROJECT_ROOT / "scripts").rglob("*.py")
            if "ArgumentParser(" in path.read_text(encoding="utf-8")
        )
    )


SCRIPTS = _argparse_scripts()


# Help text of the original scripts, captured at 80 columns before the refactor.
CHANGELOG_HELP = """\
usage: changelog.py [-h] [--changelog CHANGELOG] tag

Extract the curated changelog section for a release tag.

positional arguments:
  tag                   release tag, for example 0.6.0 or 0.6.0rc1

options:
  -h, --help            show this help message and exit
  --changelog CHANGELOG
"""
ARCHITECTURE_HELP = """\
usage: check_architecture.py [-h] [--package PACKAGE]

Check import cycles and dependency boundaries in the package.

options:
  -h, --help         show this help message and exit
  --package PACKAGE
"""


def _run_script(
    script: Path, args: list[str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Run a script by its path, the way the release workflow and gauntlet do."""
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        check=False,
        cwd=cwd,
        env={**os.environ, "NO_COLOR": "1"},
        text=True,
    )


def _run_module(module: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a script as ``python -m`` from the repository root."""
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        check=False,
        cwd=PROJECT_ROOT,
        env={**os.environ, "NO_COLOR": "1"},
        text=True,
    )


def _write_package(root: Path, modules: dict[str, str]) -> Path:
    """Create a tiny package named ``geoparser`` with the given modules."""
    package = root / "geoparser"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name, source in modules.items():
        path = package / f"{name.replace('.', '/')}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return package


@pytest.fixture
def changelog_file(tmp_path: Path) -> Path:
    path = tmp_path / "CHANGELOG.md"
    path.write_text(CHANGELOG_TEXT, encoding="utf-8")
    return path


# --- scripts/changelog.py ---------------------------------------------------


def test_changelog_runs_directly_from_outside_the_repository(
    tmp_path: Path, changelog_file: Path
) -> None:
    result = _run_script(
        CHANGELOG_SCRIPT,
        ["1.0.0rc1", "--changelog", str(changelog_file)],
        cwd=tmp_path,
    )

    assert (result.returncode, result.stdout, result.stderr) == (0, "Shipped.\n", "")


def test_changelog_runs_as_a_module_from_the_repository_root(
    changelog_file: Path,
) -> None:
    result = _run_module(
        "scripts.changelog", ["1.0.0", "--changelog", str(changelog_file)]
    )

    assert (result.returncode, result.stdout, result.stderr) == (0, "Shipped.\n", "")


def test_changelog_prints_the_stable_section_for_a_release_candidate(
    capsys: pytest.CaptureFixture[str], changelog_file: Path
) -> None:
    exit_code = changelog.main(["1.0.0rc3", "--changelog", str(changelog_file)])

    assert exit_code == 0
    assert capsys.readouterr() == ("Shipped.\n", "")


def test_changelog_reports_a_missing_section_on_stderr_only(
    capsys: pytest.CaptureFixture[str], changelog_file: Path
) -> None:
    exit_code = changelog.main(["2.0.0", "--changelog", str(changelog_file)])

    assert exit_code == 1
    assert capsys.readouterr() == ("", "No changelog section for 2.0.0\n")


def test_changelog_reports_an_empty_section_on_stderr_only(
    capsys: pytest.CaptureFixture[str], changelog_file: Path
) -> None:
    exit_code = changelog.main(["0.9.0", "--changelog", str(changelog_file)])

    assert exit_code == 1
    assert capsys.readouterr() == ("", "Changelog section for 0.9.0 is empty\n")


def test_changelog_reports_a_missing_file_on_stderr_only(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    missing = tmp_path / "absent.md"

    exit_code = changelog.main(["1.0.0", "--changelog", str(missing)])

    assert exit_code == 1
    assert capsys.readouterr() == (
        "",
        f"[Errno 2] No such file or directory: {str(missing)!r}\n",
    )


def test_changelog_defaults_to_changelog_md_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    first = (changelog.main(["1.0.0"]), capsys.readouterr().err)
    assert first == (1, "[Errno 2] No such file or directory: 'CHANGELOG.md'\n")

    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG_TEXT, encoding="utf-8")

    second = (changelog.main(["1.0.0"]), capsys.readouterr().out)
    assert second == (0, "Shipped.\n")


def test_changelog_requires_a_tag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        changelog.main([])

    assert error.value.code == 2
    assert "required: tag" in capsys.readouterr().err


def test_changelog_rejects_unknown_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        changelog.main(["1.0.0", "--bogus"])

    assert error.value.code == 2
    assert "unrecognized arguments: --bogus" in capsys.readouterr().err


def test_changelog_help_matches_the_golden_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The program name is fixed here: Python 3.14 derives it from how it was run."""
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("NO_COLOR", "1")
    parser = changelog.build_parser()
    parser.prog = "changelog.py"

    assert parser.format_help() == CHANGELOG_HELP


@pytest.mark.parametrize(
    ("script", "usage"),
    [
        (CHANGELOG_SCRIPT, "usage: changelog.py"),
        (ARCHITECTURE_SCRIPT, "usage: check_architecture.py"),
    ],
)
def test_help_runs_directly_from_outside_the_repository(
    script: Path, usage: str, tmp_path: Path
) -> None:
    result = _run_script(script, ["--help"], cwd=tmp_path)

    assert (result.returncode, result.stderr) == (0, "")
    assert usage in result.stdout


@pytest.mark.parametrize(
    ("module", "script"),
    [
        ("scripts.changelog", "changelog.py"),
        ("scripts.check_architecture", "check_architecture.py"),
    ],
)
def test_help_runs_as_a_module_from_the_repository_root(
    module: str, script: str
) -> None:
    """
    Name the program by its module or by its file, whichever Python prints.

    Python 3.14 prints ``<interpreter> -m <module>``; older versions print the
    file name. The program is compared as a whole token, so a similar name such
    as ``my_changelog.py`` or ``scripts.changelog_old`` does not pass.
    """
    result = _run_module(module, ["--help"])
    tokens = result.stdout.splitlines()[0].split()

    assert (result.returncode, result.stderr) == (0, "")
    assert result.stdout.startswith("usage: ")
    assert tokens[1] == script or tokens[2:4] == ["-m", module]


# --- scripts/check_architecture.py ------------------------------------------


def test_architecture_runs_directly_from_outside_the_repository(
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path, {"db": "VALUE = 1\n"})

    result = _run_script(ARCHITECTURE_SCRIPT, ["--package", str(package)], cwd=tmp_path)

    assert (result.returncode, result.stdout, result.stderr) == (
        0,
        "Architecture checks passed for geoparser (2 modules).\n",
        "",
    )


def test_architecture_runs_as_a_module_from_the_repository_root(
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path, {"db": "VALUE = 1\n"})

    result = _run_module("scripts.check_architecture", ["--package", str(package)])

    assert result.returncode == 0
    assert result.stdout == "Architecture checks passed for geoparser (2 modules).\n"
    assert result.stderr == ""


def test_architecture_checks_the_package_in_the_working_directory_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_package(tmp_path, {"db": "VALUE = 1\n"})
    monkeypatch.chdir(tmp_path)

    assert check_architecture.main([]) == 0
    assert capsys.readouterr() == (
        "Architecture checks passed for geoparser (2 modules).\n",
        "",
    )


def test_architecture_reports_forbidden_edges_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_package(
        tmp_path,
        {
            "db": "from geoparser.context import Thing\n",
            "context": "Thing = 1\n",
        },
    )

    exit_code = check_architecture.main(["--package", str(package)])

    assert exit_code == 1
    assert capsys.readouterr() == (
        "Forbidden dependency edges:\n  geoparser.db -> geoparser.context\n",
        "",
    )


def test_architecture_reports_import_cycles_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_package(
        tmp_path,
        {"a": "import geoparser.b\n", "b": "import geoparser.a\n"},
    )

    exit_code = check_architecture.main(["--package", str(package)])

    assert exit_code == 1
    assert capsys.readouterr() == (
        "Import cycles:\n  geoparser.a -> geoparser.b -> geoparser.a\n",
        "",
    )


def test_architecture_reports_impure_pure_modules_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_package(tmp_path, {"pure": "import torch\n"})
    monkeypatch.setattr(check_architecture, "PURE_MODULES", {"geoparser.pure"})

    exit_code = check_architecture.main(["--package", str(package)])

    assert exit_code == 1
    assert capsys.readouterr() == (
        "Pure modules with a forbidden dependency:\n  geoparser.pure -> torch\n",
        "",
    )


def test_architecture_scans_a_missing_path_as_an_empty_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pins current behaviour: a path that does not exist passes with zero modules.

    Changing this is a deliberate behaviour change and should update this test
    in the same commit.
    """
    missing = tmp_path / "geoparser"

    exit_code = check_architecture.main(["--package", str(missing)])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "Architecture checks passed for geoparser (0 modules).\n"
    )


def test_architecture_rejects_unknown_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        check_architecture.main(["--bogus"])

    assert error.value.code == 2
    assert "unrecognized arguments: --bogus" in capsys.readouterr().err


def test_architecture_help_matches_the_golden_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("NO_COLOR", "1")
    parser = check_architecture.build_parser()
    parser.prog = "check_architecture.py"

    assert parser.format_help() == ARCHITECTURE_HELP


# --- build_parser() ---------------------------------------------------------


@pytest.mark.parametrize(
    ("build_parser", "argv", "expected"),
    [
        (
            changelog.build_parser,
            ["1.0.0"],
            {"tag": "1.0.0", "changelog": Path("CHANGELOG.md")},
        ),
        (
            changelog.build_parser,
            ["0.6.0rc1", "--changelog", "x.md"],
            {"tag": "0.6.0rc1", "changelog": Path("x.md")},
        ),
        (
            changelog.build_parser,
            ["--changelog=x.md", "1.0.0"],
            {"tag": "1.0.0", "changelog": Path("x.md")},
        ),
        (check_architecture.build_parser, [], {"package": Path("geoparser")}),
        (
            check_architecture.build_parser,
            ["--package", "pkg"],
            {"package": Path("pkg")},
        ),
    ],
)
def test_parsers_produce_the_documented_values(
    build_parser: Callable[[], argparse.ArgumentParser],
    argv: list[str],
    expected: dict[str, object],
) -> None:
    assert vars(build_parser().parse_args(argv)) == expected


@contextmanager
def _filesystem_access_is_refused() -> Iterator[None]:
    """Fail if anything in the block reaches a common file entry point.

    The patches are undone when the block exits, before pytest formats a failure,
    so the report is not itself broken by the refusals.
    """

    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "parser construction touched the filesystem"
        raise AssertionError(msg)

    targets: list[tuple[object, str]] = [
        (Path, name) for name in ("read_text", "resolve", "rglob", "iterdir", "exists")
    ]
    targets.append((builtins, "open"))
    with ExitStack() as stack:
        for owner, name in targets:
            stack.enter_context(patch.object(owner, name, refuse))
        yield


def test_building_either_parser_touches_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG_TEXT, encoding="utf-8")
    before = sorted(os.listdir(tmp_path))

    with _filesystem_access_is_refused():
        changelog.build_parser().parse_args(["1.0.0"])
        check_architecture.build_parser().parse_args([])

    assert sorted(os.listdir(tmp_path)) == before


@pytest.mark.parametrize(
    ("build_parser", "description"),
    [
        (changelog.build_parser, changelog.__doc__),
        (check_architecture.build_parser, check_architecture.__doc__),
    ],
)
def test_build_parser_keeps_the_module_docstring_as_its_description(
    build_parser: Callable[[], argparse.ArgumentParser], description: str | None
) -> None:
    assert build_parser().description == description


def _parse_script(script: Path) -> ast.Module:
    return ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


def _top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _is_parser_call(node: ast.AST) -> bool:
    """Whether a node calls ``ArgumentParser``, as ``argparse.`` or imported bare."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr == "ArgumentParser") or (
        isinstance(func, ast.Name) and func.id == "ArgumentParser"
    )


def _parser_constructions(node: ast.AST) -> int:
    """Count the ``ArgumentParser(...)`` calls beneath a node."""
    return sum(_is_parser_call(call) for call in ast.walk(node))


def _calls_build_parser(node: ast.AST) -> bool:
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "build_parser"
        for call in ast.walk(node)
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_only_build_parser_constructs_the_argument_parser(script: Path) -> None:
    tree = _parse_script(script)
    functions = _top_level_functions(tree)

    assert {"build_parser", "main"} <= functions.keys()
    assert (
        _parser_constructions(functions["build_parser"])
        == _parser_constructions(tree)
        == 1
    )
    assert _calls_build_parser(functions["main"])


@pytest.mark.parametrize("script", SCRIPTS)
def test_main_takes_an_optional_argument_list(script: Path) -> None:
    main = _top_level_functions(_parse_script(script))["main"]

    assert [argument.arg for argument in main.args.args] == ["argv"]
    assert len(main.args.defaults) == 1
