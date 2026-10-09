"""Offline checks on the DaMuEL release record and its canonical-language coverage.

The expected language sets below are written out by hand from the public LINDAT
listing. They are an oracle independent of the code that computes coverage, so a
silent change to the snapshot or to the matching rule fails here.
"""

import copy
import json

import pytest
from pydantic import ValidationError

from scripts.damuel_inventory import __main__ as cli
from scripts.damuel_inventory.inventory import (
    RELEASE_PATH,
    Release,
    coverage,
    coverage_summary,
    load_release,
)

COVERED = {
    "af",
    "ar",
    "be",
    "bg",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "ga",
    "gd",
    "gl",
    "he",
    "hi",
    "hu",
    "hy",
    "id",
    "it",
    "ja",
    "ko",
    "la",
    "lt",
    "lv",
    "mr",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    "ta",
    "te",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh",
}
MISSING = {
    "am",
    "az",
    "bn",
    "ceb",
    "cy",
    "eo",
    "fy",
    "gu",
    "ha",
    "ig",
    "is",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ku",
    "ky",
    "mg",
    "mk",
    "ml",
    "mn",
    "ms",
    "my",
    "ne",
    "no",
    "pa",
    "ps",
    "si",
    "sq",
    "tg",
    "th",
    "uz",
    "xh",
    "yi",
    "yo",
    "zu",
}
RELEASE_ONLY = {"hr", "nn", "se", "ug", "wo"}


def release_payload():
    """A deep copy of the checked-in record, safe to mutate in a test."""
    return copy.deepcopy(json.loads(RELEASE_PATH.read_text(encoding="utf-8")))


def test_checked_in_record_validates_and_names_the_expected_archives():
    release = load_release()
    assert isinstance(release, Release)
    assert len(release.files) == 54
    knowledge_bases = [
        entry for entry in release.files if entry.kind == "knowledge_base"
    ]
    assert [entry.file for entry in knowledge_bases] == ["damuel_1.0_wikidata.tar"]
    assert knowledge_bases[0].size_bytes == 2_715_955_200
    assert knowledge_bases[0].md5 == "778ccad6e829d59938419064c9f8de4d"


def test_byte_sizes_and_checksums_match_the_two_values_read_from_the_api():
    by_name = {entry.file: entry for entry in load_release().files}
    wiki = by_name["damuel_1.0_wo.tar"]
    assert (wiki.size_bytes, wiki.md5) == (
        10_721_280,
        "bc04313f6d6727963ebc77d5a353b7a0",
    )


def test_licence_is_recorded_with_its_verified_source_only():
    licence = load_release().licence
    assert licence.name.endswith("(CC BY-SA 4.0)")
    assert any("LICENSE member" in source for source in licence.verified_sources)
    assert any("Wikipedia" in item for item in licence.not_verified)


def test_coverage_matches_the_hand_counted_language_sets():
    report = coverage(load_release())
    assert len(report.canonical) == 85
    assert (len(report.covered), len(report.missing), len(report.release_only)) == (
        48,
        37,
        5,
    )
    assert set(report.covered) == COVERED
    assert set(report.missing) == MISSING
    assert set(report.release_only) == RELEASE_ONLY


def test_matching_is_exact_so_norwegian_bokmal_is_not_read_from_nynorsk():
    report = coverage(load_release())
    assert "no" in report.missing
    assert "nn" in report.release_only
    assert "no" not in report.covered


def test_targets_argument_restricts_the_comparison_without_touching_the_record():
    report = coverage(load_release(), ("en", "fr", "xh", "wo"))
    assert report.covered == ("en", "fr", "wo")
    assert report.missing == ("xh",)
    assert "ug" in report.release_only
    assert "en" not in report.release_only


def test_summary_counts_agree_with_their_listed_members():
    summary = coverage_summary(load_release())
    assert summary["covered_count"] == len(summary["covered"]) == 48
    assert summary["missing_count"] == len(summary["missing"]) == 37
    assert summary["release_only_count"] == len(summary["release_only"]) == 5
    assert summary["canonical_target_count"] == 85


def test_archive_name_must_match_its_language_code():
    payload = release_payload()
    payload["files"][1]["language"] = "xx"
    with pytest.raises(ValidationError, match="must match its language code"):
        Release.model_validate(payload)


def test_knowledge_base_must_use_the_wikidata_archive_name():
    payload = release_payload()
    kb = next(entry for entry in payload["files"] if entry["kind"] == "knowledge_base")
    kb["file"] = "damuel_1.0_en.tar"
    with pytest.raises(ValidationError):
        Release.model_validate(payload)


def test_duplicate_archive_name_is_rejected():
    payload = release_payload()
    payload["files"][2] = copy.deepcopy(payload["files"][1])
    with pytest.raises(ValidationError, match="more than once"):
        Release.model_validate(payload)


def test_missing_text_archive_is_rejected():
    payload = release_payload()
    text_index = next(
        index for index, entry in enumerate(payload["files"]) if entry["kind"] == "text"
    )
    del payload["files"][text_index]
    with pytest.raises(ValidationError, match="53 text archives"):
        Release.model_validate(payload)


def test_impossible_calendar_day_is_rejected():
    payload = release_payload()
    payload["retrieved_on"] = "2026-02-30"
    with pytest.raises(ValidationError, match="real calendar day"):
        Release.model_validate(payload)


def test_unknown_field_is_rejected_rather_than_silently_kept():
    payload = release_payload()
    payload["files"][0]["sample_rows"] = 100
    with pytest.raises(ValidationError):
        Release.model_validate(payload)


def test_cli_prints_the_coverage_report(capsys):
    assert cli.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result["covered_count"], result["missing_count"]) == (48, 37)
    assert result["licence"].endswith("(CC BY-SA 4.0)")


def test_cli_reports_a_malformed_record_as_an_error(tmp_path, capsys):
    path = tmp_path / "release.json"
    path.write_text("{not json", encoding="utf-8")
    assert cli.main(["--release", str(path)]) == 2
    assert "Invalid DaMuEL release record" in capsys.readouterr().err


def test_cli_reports_a_missing_record_as_an_error(tmp_path, capsys):
    assert cli.main(["--release", str(tmp_path / "absent.json")]) == 2
    assert "absent.json" in capsys.readouterr().err
