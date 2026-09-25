"""
End-to-end CliRunner tests for the ``geoparser`` command.

These drive the real Typer app, so they check exit codes and which stream a
message lands on, not just that the underlying function was called.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from geoparser.cli.app import app

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

runner = CliRunner()


@pytest.mark.unit
class TestTopLevel:
    """Test options that belong to the application itself."""

    def test_version_prints_package_version(self):
        from importlib.metadata import version

        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert result.stdout.strip() == version("geoparser")

    def test_no_arguments_shows_help(self):
        result = runner.invoke(app, [])

        # Click exits 2 for "missing command" while still printing the help.
        assert "Usage" in result.output
        assert "install" in result.output

    def test_console_script_is_declared(self):
        pyproject = tomllib.loads(
            (Path(__file__).parents[3] / "pyproject.toml").read_text()
        )

        scripts = pyproject["project"]["scripts"]
        assert scripts["geoparser"] == "geoparser.__main__:main"


@pytest.mark.unit
class TestInstallErrors:
    """Test that user mistakes produce messages, not tracebacks."""

    @patch("geoparser.cli.install._get_builtin_gazetteers")
    def test_unknown_name_exits_2_without_traceback(self, mock_builtin):
        mock_builtin.return_value = {"geonames": Path("/builtin/geonames.yaml")}

        result = runner.invoke(app, ["install", "nope"])

        assert result.exit_code == 2
        assert "Available built-in gazetteer configs" in result.stderr
        assert "geonames" in result.stderr
        assert "Traceback" not in result.output
        assert not isinstance(result.exception, FileNotFoundError)

    @patch("geoparser.gazetteer.build.GazetteerBuilder")
    def test_build_failure_prints_one_line_and_exits_1(self, mock_builder, tmp_path):
        config = tmp_path / "custom.yaml"
        config.write_text("name: custom\n")
        mock_builder.return_value.build.side_effect = OSError("No space left")

        result = runner.invoke(app, ["install", str(config)])

        assert result.exit_code == 1
        assert "No space left" in result.stderr
        assert "--verbose" in result.stderr

    @patch("geoparser.gazetteer.build.GazetteerBuilder")
    def test_build_failure_with_verbose_reraises(self, mock_builder, tmp_path):
        config = tmp_path / "custom.yaml"
        config.write_text("name: custom\n")
        mock_builder.return_value.build.side_effect = OSError("No space left")

        result = runner.invoke(app, ["install", str(config), "--verbose"])

        assert result.exit_code == 1
        assert isinstance(result.exception, OSError)
