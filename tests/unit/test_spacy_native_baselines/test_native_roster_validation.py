"""Roster validation: each invariant rejects a malformed document.

Every case starts from the checked-in roster and changes one field, so the
expected error is the only invariant that the change breaks.
"""

import json

import pytest

from scripts.spacy_native_baselines.roster import (
    ROSTER_PATH,
    RosterError,
    load_roster,
    parse_roster,
)


@pytest.fixture
def document():
    return json.loads(ROSTER_PATH.read_text(encoding="utf-8"))


def _entry(document, language):
    return document["languages"][language]


def _unsupported_code(document):
    return next(
        code
        for code, entry in document["languages"].items()
        if entry["status"] == "unsupported"
    )


def test_checked_in_roster_parses(document):
    assert parse_roster(document).languages["de"] == "native"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("format", "other-format", "Roster format must be"),
        ("model_version", "3.7.0", "pin spaCy model version 3.8.0"),
    ],
)
def test_header_fields_are_checked(document, key, value, message):
    document[key] = value

    with pytest.raises(RosterError, match=message):
        parse_roster(document)


def test_languages_must_be_an_object(document):
    document["languages"] = []

    with pytest.raises(RosterError, match="languages object"):
        parse_roster(document)


def test_languages_must_be_the_target_codes_in_order(document):
    document["languages"] = dict(reversed(list(document["languages"].items())))

    with pytest.raises(RosterError, match="85 target codes, in order"):
        parse_roster(document)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("version", "3.7.0", "wheel is not the pinned 3.8.0 release"),
        ("wheel_url", "https://example.invalid/x.whl", "name its release tag"),
        ("spacy_language", "xx", "does not belong to xx"),
        ("place_label_map", {"NOT_A_LABEL": "LOC"}, "not in the pipeline's NER labels"),
        ("place_label_map", {"LOC": "CITY"}, "must map only to LOC"),
    ],
)
def test_native_entries_are_checked(document, key, value, message):
    _entry(document, "de")[key] = value

    with pytest.raises(RosterError, match=message):
        parse_roster(document)


def test_unsupported_entries_need_a_reason(document):
    _entry(document, _unsupported_code(document))["reason"] = ""

    with pytest.raises(RosterError, match="need a reason"):
        parse_roster(document)


def test_unknown_status_is_rejected(document):
    _entry(document, "de")["status"] = "maybe"

    with pytest.raises(RosterError, match="unknown status 'maybe'"):
        parse_roster(document)


def test_english_control_must_be_the_english_pipeline_package(document):
    document["english_control"]["package"] = "de_core_news_sm"

    with pytest.raises(RosterError, match="English control"):
        parse_roster(document)


def test_english_control_must_be_a_native_english_entry(document):
    _entry(document, "en")["status"] = "unsupported"
    _entry(document, "en")["reason"] = "withdrawn for this test"

    with pytest.raises(RosterError, match="English control"):
        parse_roster(document)


def test_roster_file_must_hold_a_json_object(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(RosterError, match="must be an object"):
        load_roster(path)
