"""Offline dry-run and local validation commands; no download is attempted."""

import json

import pytest

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


@pytest.fixture
def parse_only(monkeypatch):
    """Parse-level tests use short fixture text, so they skip the pinned check."""
    monkeypatch.setattr(
        "scripts.multiconer_benchmark.__main__.check_local_file", lambda *args: None
    )


def _validate_argv(path) -> list[str]:
    return ["--validate-conll", str(path), "--language", "en", "--split", "test"]


def _validate_clean_file(tmp_path, capsys) -> dict:
    path = tmp_path / "en_test.conll"
    path.write_text(
        "# id one\tdomain=en\nParis _ _ B-HumanSettlement\n", encoding="utf-8"
    )
    assert main(_validate_argv(path)) == 0
    return json.loads(capsys.readouterr().out)


def test_clean_local_file_validates_with_exit_zero(tmp_path, capsys, parse_only):
    report = _validate_clean_file(tmp_path, capsys)
    assert report["records"] == 1
    assert report["valid"] == 1
    assert report["invalid"] == 0


def test_clean_local_file_reports_its_location_spans(tmp_path, capsys, parse_only):
    report = _validate_clean_file(tmp_path, capsys)
    assert report["location_spans"] == 1


def test_invalid_local_records_are_reported_and_fail_the_command(
    tmp_path, capsys, parse_only
):
    path = tmp_path / "en_test.conll"
    path.write_text("Rome _ _ B-HumanSettlement\n\n", encoding="utf-8")
    assert main(_validate_argv(path)) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["invalid"] == 1
    assert report["invalid_reasons"] == {"missing sentence header": 1}


def test_unreadable_or_non_utf8_file_is_an_error_not_a_zero_count(
    tmp_path, capsys, parse_only
):
    path = tmp_path / "bad.conll"
    path.write_bytes(b"\xff\xfe# id x\n")
    assert main(_validate_argv(path)) == 2
    assert "Invalid" in capsys.readouterr().err


def test_a_file_that_differs_from_the_pinned_split_is_refused(tmp_path, capsys):
    path = tmp_path / "en_test.conll"
    path.write_text(
        "# id one\tdomain=en\nParis _ _ B-HumanSettlement\n", encoding="utf-8"
    )
    assert main(_validate_argv(path)) == 2
    assert "pinned en test file" in capsys.readouterr().err


def test_validation_needs_both_the_language_and_the_split(tmp_path):
    path = tmp_path / "en_test.conll"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["--validate-conll", str(path), "--language", "en"])
    assert error.value.code == 2
