from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from typer.testing import CliRunner

from geoparser.cli.app import app
from geoparser.gazetteer.gazetteer import Gazetteer

pytestmark = pytest.mark.acceptance
scenarios("features/gazetteer_install.feature")


@pytest.fixture
def cli_state(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    monkeypatch.setenv("GEOPARSER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("GEOPARSER_GAZETTEERS_DIR", raising=False)
    return {}


@given("a gazetteer config pointing at the local Andorra fixture")
def local_config(cli_state: dict[str, Any], andorra_config_path: Path) -> None:
    cli_state["argument"] = str(andorra_config_path)


def _run(cli_state: dict[str, Any], command: str, argument: str) -> None:
    cli_state["result"] = CliRunner().invoke(app, [command, argument])


@when(parsers.parse('I run the geoparser command "{command}" with that config'))
def run_with_config(cli_state: dict[str, Any], command: str) -> None:
    _run(cli_state, command, cli_state["argument"])


@when(parsers.parse('I run the geoparser command "{command}" with "{argument}"'))
def run_with_argument(cli_state: dict[str, Any], command: str, argument: str) -> None:
    _run(cli_state, command, argument)


@then(parsers.parse("the exit code is {code:d}"))
def exit_code(cli_state: dict[str, Any], code: int) -> None:
    result = cli_state["result"]
    assert result.exit_code == code, result.output


@then(parsers.parse('the installed gazetteer finds "{name}"'))
def gazetteer_finds(name: str) -> None:
    features = Gazetteer("andorranames").search(name, method="exact")
    assert any(name in (feature.data or {}).values() for feature in features)


@then(parsers.parse('the error output names "{name}" without a traceback'))
def clean_error(cli_state: dict[str, Any], name: str) -> None:
    result = cli_state["result"]
    assert name in result.stderr
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
