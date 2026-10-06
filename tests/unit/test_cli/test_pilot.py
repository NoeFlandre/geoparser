from pathlib import Path

import pytest

from scripts import pilot


@pytest.fixture
def clean_pilot_runtime(monkeypatch):
    """Start with no inherited database or Hugging Face runtime settings."""
    for name in ("DATABASE_URL", "GEOPARSER_GAZETTEERS_DIR", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)


def test_runtime_configuration_routes_application_artifacts(
    clean_pilot_runtime, tmp_path: Path
) -> None:
    output_dir = tmp_path / "output"

    pilot._configure_runtime(output_dir, None, offline=False)

    assert (
        pilot.os.environ["DATABASE_URL"] == f"sqlite:///{output_dir / 'pilot.sqlite'}"
    )
    assert pilot.os.environ["GEOPARSER_GAZETTEERS_DIR"] == str(
        output_dir / "gazetteers"
    )
    assert output_dir.is_dir()


def test_runtime_configuration_uses_requested_hub_cache(
    clean_pilot_runtime, tmp_path: Path
) -> None:
    hf_home = tmp_path / "hub"

    pilot._configure_runtime(tmp_path / "output", hf_home, offline=False)

    assert pilot.os.environ["HF_HOME"] == str(hf_home)


@pytest.mark.parametrize(("offline", "expected"), [(False, None), (True, "1")])
def test_runtime_configuration_sets_offline_switches(
    clean_pilot_runtime, tmp_path: Path, offline: bool, expected: str | None
) -> None:
    pilot._configure_runtime(tmp_path / "output", None, offline=offline)

    assert pilot.os.environ.get("HF_HUB_OFFLINE") == expected
    assert pilot.os.environ.get("TRANSFORMERS_OFFLINE") == expected


def test_pilot_cli_passes_explicit_paths_and_prints_both_reports(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    calls = []
    reports = (tmp_path / "result.json", tmp_path / "result.md")

    def run_pilot(**kwargs):
        calls.append(kwargs)
        return reports

    monkeypatch.setattr(pilot, "run_pilot", run_pilot)

    exit_code = pilot.main(
        [
            "--config",
            str(tmp_path / "andorra.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--hf-home",
            str(tmp_path / "hub"),
            "--offline",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "config_path": tmp_path / "andorra.yaml",
            "output_dir": tmp_path / "out",
            "hf_home": tmp_path / "hub",
            "offline": True,
        }
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"JSON report: {reports[0]}",
        f"Markdown report: {reports[1]}",
    ]
