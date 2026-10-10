"""The offline command line: a registry dump, and a freeze that writes only on success."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.benchmark_protocol.schema import Experiment
from scripts.embedding_resolution import __main__ as cli
from scripts.embedding_resolution.measurement import peak_rss_bytes, timed
from tests.unit.test_embedding_resolution.test_protocol import plan_payload


def test_registry_prints_every_pin_and_scoring_constant(capsys):
    assert cli.main(["registry"]) == 0

    document = json.loads(capsys.readouterr().out)
    assert (
        document["verified_on"],
        [model["key"] for model in document["models"]],
        document["population_weight"],
        document["distance_thresholds_km"],
        document["calibration_grid"]["points"],
    ) == (
        "2026-10-09",
        [
            "geo-minilm",
            "jina-v5-text-small",
            "qwen3-embedding-0.6b",
            "qwen3-embedding-4b",
            "bge-m3",
        ],
        0.3,
        [1.0, 10.0, 50.0],
        201,
    )


def test_freeze_writes_the_validated_inventory_and_prints_its_summary(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_payload()), encoding="utf-8")
    output = tmp_path / "experiment.json"

    assert cli.main(["freeze", str(plan), "--output", str(output)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["configuration_count"] == 6
    assert summary["status_counts"] == {"planned": 6}
    experiment = Experiment.model_validate_json(output.read_text(encoding="utf-8"))
    assert len(experiment.configurations) == 6


def test_an_invalid_plan_exits_two_and_writes_nothing(tmp_path, capsys):
    payload = plan_payload()
    payload["protocol"]["task"] = "recognition"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "experiment.json"

    assert cli.main(["freeze", str(plan), "--output", str(output)]) == 2

    assert "Invalid freeze plan" in capsys.readouterr().err
    assert not output.exists()


def test_a_missing_plan_file_exits_two(tmp_path, capsys):
    assert cli.main(["freeze", str(tmp_path / "absent.json")]) == 2

    assert "Invalid freeze plan" in capsys.readouterr().err


def test_a_plan_that_is_not_utf8_exits_two(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_bytes(b"\xff\xfe\x00")

    assert cli.main(["freeze", str(plan)]) == 2


def test_the_module_entry_point_runs_the_registry_command():
    root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [sys.executable, "-m", "scripts.embedding_resolution", "registry"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["verified_on"] == "2026-10-09"


def test_timed_returns_the_result_and_a_non_negative_duration():
    result, seconds = timed(lambda: 6 * 7)

    assert result == 42
    assert seconds >= 0.0


def test_peak_rss_is_converted_to_bytes_on_macos_and_kibibytes_elsewhere(monkeypatch):
    # The resource module is POSIX-only, so the unit test is skipped on Windows.
    resource = pytest.importorskip(
        "resource", reason="peak RSS is read with the POSIX resource module"
    )

    class Usage:
        ru_maxrss = 2048

    monkeypatch.setattr(resource, "getrusage", lambda _who: Usage())

    monkeypatch.setattr(sys, "platform", "darwin")
    assert peak_rss_bytes() == 2048
    monkeypatch.setattr(sys, "platform", "linux")
    assert peak_rss_bytes() == 2048 * 1024


def test_the_documented_example_plan_is_valid():
    root = Path(__file__).resolve().parents[3]
    example = root / "docs" / "examples" / "embedding-resolution-plan.json"

    assert cli.main(["freeze", str(example)]) == 0
