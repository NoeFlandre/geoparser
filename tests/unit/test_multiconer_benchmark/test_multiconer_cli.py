"""Offline dry-run and local validation commands; no download is attempted."""

import json

from scripts.multiconer_benchmark.__main__ import main


def _dry_run_report(capsys) -> dict:
    assert main(["--manifest"]) == 0
    return json.loads(capsys.readouterr().out)


def test_dry_run_prints_the_pinned_inventory_without_network(capsys):
    report = _dry_run_report(capsys)
    assert report["dataset"]["revision"] == "4be2d62c912977ee26ed14d2553a4fe17ca3d980"
    assert report["languages"] == [
        "bn",
        "de",
        "en",
        "es",
        "fa",
        "fr",
        "hi",
        "it",
        "pt",
        "sv",
        "uk",
        "zh",
    ]
    assert report["excluded"] == ["multi"]
    assert report["gap_count"] == 6


def _validate_clean_file(tmp_path, capsys) -> dict:
    path = tmp_path / "en_test.conll"
    path.write_text(
        "# id one\tdomain=en\nParis _ _ B-HumanSettlement\n", encoding="utf-8"
    )
    assert main(["--validate-conll", str(path), "--language", "en"]) == 0
    return json.loads(capsys.readouterr().out)


def test_clean_local_file_validates_with_exit_zero(tmp_path, capsys):
    report = _validate_clean_file(tmp_path, capsys)
    assert report["records"] == 1
    assert report["valid"] == 1
    assert report["invalid"] == 0


def test_clean_local_file_reports_its_location_spans(tmp_path, capsys):
    report = _validate_clean_file(tmp_path, capsys)
    assert report["location_spans"] == 1


def test_invalid_local_records_are_reported_and_fail_the_command(tmp_path, capsys):
    path = tmp_path / "en_test.conll"
    path.write_text("Rome _ _ B-HumanSettlement\n\n", encoding="utf-8")
    assert main(["--validate-conll", str(path), "--language", "en"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["invalid"] == 1
    assert report["invalid_reasons"] == {"missing sentence header": 1}


def test_unreadable_or_non_utf8_file_is_an_error_not_a_zero_count(tmp_path, capsys):
    path = tmp_path / "bad.conll"
    path.write_bytes(b"\xff\xfe# id x\n")
    assert main(["--validate-conll", str(path), "--language", "en"]) == 2
    assert "Invalid" in capsys.readouterr().err
