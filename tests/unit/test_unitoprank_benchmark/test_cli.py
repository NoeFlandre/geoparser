"""The checker reports the pin and exits non-zero on a mismatch, without ranking."""

import json
from pathlib import Path

from scripts.unitoprank_benchmark import __main__ as cli
from scripts.unitoprank_benchmark.pins import COMMIT


def test_an_empty_directory_exits_two_and_reports_the_missing_files(
    tmp_path: Path, capsys
):
    assert cli.main(["--checkout", str(tmp_path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["problems"][0] == "missing: LICENSE"


def test_the_report_names_the_pin_and_how_many_files_it_covers(tmp_path: Path, capsys):
    cli.main(["--checkout", str(tmp_path)])
    report = json.loads(capsys.readouterr().out)
    assert report["commit"] == COMMIT
    assert report["license"] == "Apache-2.0"
    assert report["files_pinned"] == 15


def test_a_matching_checkout_exits_zero(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "verify_checkout", lambda _path: [])
    assert cli.main(["--checkout", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["problems"] == []
