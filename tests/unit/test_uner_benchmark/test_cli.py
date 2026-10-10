"""Exercise the actual download-free command boundary."""

import json

import pytest

from scripts.uner_benchmark import __main__ as cli
from tests.unit.test_uner_benchmark.test_inventory import PAYLOAD, dataset


@pytest.fixture
def registry(monkeypatch):
    specs = (
        dataset(),
        dataset(configuration="sv_fixture", source_language="sv", target_language="sv"),
    )
    monkeypatch.setattr(cli, "read_manifest", lambda: specs)
    return specs


def test_registry_dry_run_reports_all_selected_configurations(registry, capsys):
    assert cli.main(["--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["selected_configurations"] == ["en_fixture", "sv_fixture"]
    assert result["validation"] == "registry_only"
    assert (result["local_sources"], result["model_evaluation"]) == ([], "not_run")


def test_explicit_configuration_and_split_are_not_broadened(registry, capsys):
    assert (
        cli.main(["--dry-run", "--configuration", "en_fixture", "--split", "test"]) == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["selected_configurations"] == ["en_fixture"]
    assert len(result["configurations"]) == 1
    assert result["split"] == "test"


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--dry-run", "--split", "all"],
        ["--dry-run", "--configuration", "missing"],
        ["--dry-run", "--configuration", "en_fixture", "--configuration", "en_fixture"],
    ],
)
def test_invalid_cli_arguments_fail(registry, args):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


def test_local_dry_run_validates_hash_spans_and_counts(registry, tmp_path, capsys):
    path = tmp_path / registry[0].repository / "en_fixture-ud-test.iob2"
    path.parent.mkdir()
    path.write_bytes(PAYLOAD)
    assert (
        cli.main(
            ["--dry-run", "--configuration", "en_fixture", "--cache-dir", str(tmp_path)]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    entry = result["local_sources"][0]
    assert (result["validation"], entry["status"]) == ("local_sources", "valid")
    assert (
        entry["sentences"],
        entry["documents"],
        entry["tokens"],
        entry["locations"],
        entry["empty_texts"],
    ) == (1, 1, 1, 1, 0)
    assert len(entry["sha256"]) == 64


def test_local_missing_sources_stay_visible_as_failures(registry, tmp_path, capsys):
    assert cli.main(["--dry-run", "--cache-dir", str(tmp_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert len(result["local_sources"]) == 2
    assert {entry["status"] for entry in result["local_sources"]} == {"failed"}
    assert {entry["error_type"] for entry in result["local_sources"]} == {
        "FileNotFoundError"
    }


def test_local_checksum_failures_are_not_swallowed(registry, tmp_path, capsys):
    path = tmp_path / registry[0].repository / "en_fixture-ud-test.iob2"
    path.parent.mkdir()
    path.write_bytes(b"corrupt")
    assert (
        cli.main(
            ["--dry-run", "--configuration", "en_fixture", "--cache-dir", str(tmp_path)]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert "checksum" in result["local_sources"][0]["reason"]


def test_requested_unavailable_split_is_not_reported_valid(registry, capsys):
    assert cli.main(["--dry-run", "--split", "dev"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["selected_configurations"] == []
    assert {entry["status"] for entry in result["configurations"]} == {
        "split_unavailable"
    }


def test_bad_manifest_is_a_cli_validation_error(monkeypatch):
    def bad_manifest():
        raise ValueError("invalid inventory")

    monkeypatch.setattr(cli, "read_manifest", bad_manifest)
    with pytest.raises(SystemExit) as error:
        cli.main(["--dry-run"])
    assert error.value.code == 2


def test_module_entrypoint_runs_the_same_dry_run(registry, monkeypatch, capsys):
    import runpy
    import sys
    from pathlib import Path

    monkeypatch.setattr(
        "scripts.uner_benchmark.inventory.read_manifest", lambda: registry
    )
    monkeypatch.setattr(sys, "argv", ["uner_benchmark", "--dry-run"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(Path(cli.__file__)), run_name="__main__")
    assert error.value.code == 0
    assert json.loads(capsys.readouterr().out)["model_evaluation"] == "not_run"


def test_explicit_selection_cannot_hide_duplicate_source_configurations(monkeypatch):
    monkeypatch.setattr(
        cli, "read_manifest", lambda: (dataset(), dataset(target_language="sv"))
    )
    with pytest.raises(SystemExit) as error:
        cli.main(["--dry-run", "--configuration", "en_fixture"])
    assert error.value.code == 2
