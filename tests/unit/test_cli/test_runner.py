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

        result = runner.invoke(app, ["install", str(config), "--force"])

        assert result.exit_code == 1
        assert "No space left" in result.stderr
        assert "--verbose" in result.stderr

    @patch("geoparser.gazetteer.build.GazetteerBuilder")
    def test_build_failure_with_verbose_reraises(self, mock_builder, tmp_path):
        config = tmp_path / "custom.yaml"
        config.write_text("name: custom\n")
        mock_builder.return_value.build.side_effect = OSError("No space left")

        result = runner.invoke(app, ["install", str(config), "--force", "--verbose"])

        assert result.exit_code == 1
        assert isinstance(result.exception, OSError)


@pytest.mark.unit
class TestAnnotatorOptions:
    """Test that annotator flags reach ``run``."""

    @patch("geoparser.annotator.app.run")
    def test_flags_are_passed_through(self, mock_run):
        result = runner.invoke(
            app,
            [
                "annotator",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--no-browser",
                "--reload",
            ],
        )

        assert result.exit_code == 0, result.output
        mock_run.assert_called_once_with(
            use_reloader=True, host="0.0.0.0", port=8080, open_browser=False
        )

    @patch("geoparser.annotator.app.run")
    def test_default_host_is_localhost(self, mock_run):
        result = runner.invoke(app, ["annotator"])

        assert result.exit_code == 0, result.output
        assert mock_run.call_args.kwargs["host"] == "127.0.0.1"


def _write_config(directory: Path, name: str = "custom") -> Path:
    config = directory / f"{name}.yaml"
    config.write_text(f"name: {name}\n")
    return config


@pytest.mark.unit
class TestInstallOptions:
    """Test install's --keep-downloads and --force."""

    @patch("geoparser.gazetteer.build.schema.GazetteerConfig.from_yaml")
    @patch("geoparser.gazetteer.build.GazetteerBuilder")
    def test_skips_installed_gazetteer_without_force(
        self, mock_builder, mock_from_yaml, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "geoparser.gazetteer.artifact.gazetteers_dir", lambda: tmp_path
        )
        mock_from_yaml.return_value.name = "custom"
        (tmp_path / "custom.db").write_bytes(b"")
        config = _write_config(tmp_path)

        result = runner.invoke(app, ["install", str(config)])

        assert result.exit_code == 0, result.output
        assert "already installed" in result.stdout
        mock_builder.return_value.build.assert_not_called()

    @patch("geoparser.gazetteer.build.schema.GazetteerConfig.from_yaml")
    @patch("geoparser.gazetteer.build.GazetteerBuilder")
    def test_builds_when_not_installed(
        self, mock_builder, mock_from_yaml, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "geoparser.gazetteer.artifact.gazetteers_dir", lambda: tmp_path
        )
        mock_from_yaml.return_value.name = "custom"
        config = _write_config(tmp_path)

        result = runner.invoke(app, ["install", str(config), "--keep-downloads"])

        assert result.exit_code == 0, result.output
        mock_builder.return_value.build.assert_called_once_with(
            config, keep_downloads=True
        )

    @patch("geoparser.gazetteer.build.GazetteerBuilder")
    def test_force_rebuilds(self, mock_builder, tmp_path):
        config = _write_config(tmp_path)

        result = runner.invoke(app, ["install", str(config), "--force"])

        assert result.exit_code == 0, result.output
        mock_builder.return_value.build.assert_called_once_with(
            config, keep_downloads=False
        )


@pytest.mark.unit
class TestUninstallConfirmation:
    """Test that uninstall asks before deleting."""

    @pytest.fixture
    def installed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "geoparser.gazetteer.artifact.gazetteers_dir", lambda: tmp_path
        )
        artifact = tmp_path / "custom.db"
        artifact.write_bytes(b"")
        return artifact

    def test_declining_keeps_the_gazetteer(self, installed):
        result = runner.invoke(app, ["uninstall", "custom"], input="n\n")

        assert result.exit_code == 1
        assert installed.exists()

    def test_confirming_removes_the_gazetteer(self, installed):
        result = runner.invoke(app, ["uninstall", "custom"], input="y\n")

        assert result.exit_code == 0, result.output
        assert not installed.exists()

    def test_yes_skips_the_prompt(self, installed):
        result = runner.invoke(app, ["uninstall", "custom", "--yes"])

        assert result.exit_code == 0, result.output
        assert "Remove gazetteer" not in result.output
        assert not installed.exists()
