"""Versioned native spaCy roster: coverage, routing, label maps and identities."""

import dataclasses

import pytest

from scripts.panx_benchmark.data import target_languages
from scripts.spacy_native_baselines.roster import (
    UnknownLanguageError,
    configuration_id,
    load_roster,
)


@pytest.fixture(scope="module")
def roster():
    return load_roster()


def test_roster_covers_every_target_language_exactly_once(roster):
    assert tuple(roster.languages) == target_languages()
    assert len(roster.languages) == 85


def test_native_pipelines_pin_the_spacy_3_8_0_release_wheels(roster):
    for pipeline in roster.pipelines.values():
        assert pipeline.version == "3.8.0"
        assert pipeline.wheel == f"{pipeline.package}-3.8.0-py3-none-any.whl"
        assert pipeline.release_tag == f"{pipeline.package}-3.8.0"
        assert pipeline.release_tag in pipeline.wheel_url
        assert pipeline.package.startswith(pipeline.spacy_language + "_")


def test_twenty_three_target_languages_have_native_pipelines(roster):
    assert len(roster.pipelines) == 23
    assert len(roster.unsupported) == 62
    assert set(roster.pipelines).isdisjoint(roster.unsupported)


def test_norwegian_target_code_routes_to_the_bokmal_pipeline(roster):
    route = roster.route("no")

    assert route.pipeline is not None
    assert route.pipeline.spacy_language == "nb"
    assert route.pipeline.package == "nb_core_news_sm"


def test_unsupported_languages_have_no_pipeline_and_never_fall_back_to_english(
    roster,
):
    for code in roster.unsupported:
        route = roster.route(code)

        assert route.pipeline is None
        assert route.reason
        assert route.pipeline is not roster.english_control


def test_codes_outside_the_target_list_are_rejected(roster):
    # Croatian has a spaCy pipeline but is not one of the 85 target codes.
    with pytest.raises(UnknownLanguageError):
        roster.route("hr")


def test_english_control_is_the_english_native_pipeline(roster):
    assert roster.english_control.package == "en_core_web_sm"
    assert roster.route("en").pipeline == roster.english_control


def test_english_place_labels_are_harmonized_to_loc(roster):
    assert roster.english_control.place_label_map == {
        "FAC": "LOC",
        "GPE": "LOC",
        "LOC": "LOC",
    }


def test_every_place_label_map_uses_native_labels_and_targets_loc_only(roster):
    for pipeline in roster.pipelines.values():
        assert set(pipeline.place_label_map) <= set(pipeline.ner_labels)
        assert set(pipeline.place_label_map.values()) <= {"LOC"}


@pytest.mark.parametrize(
    ("code", "label", "expected"),
    [
        ("no", "GPE_LOC", "LOC"),
        ("no", "GPE_ORG", None),
        ("pl", "placeName", "LOC"),
        ("pl", "orgName", None),
        ("ro", "FACILITY", "LOC"),
        ("ko", "LC", "LOC"),
        ("ko", "OG", None),
        ("sv", "LOC", "LOC"),
        ("sv", "ORG", None),
    ],
)
def test_native_labels_map_to_loc_or_are_dropped(roster, code, label, expected):
    assert roster.pipelines[code].harmonized_label(label) == expected


def test_native_wheel_digests_come_from_a_verified_download(roster):
    assert roster.verification["wheels_downloaded"] is True
    assert roster.verification["sha256_verified"] is True
    native = [pipeline for pipeline in roster.pipelines.values() if pipeline.sha256]
    assert len(native) == 23
    assert all(len(pipeline.sha256) == 64 for pipeline in native)


def test_japanese_pipeline_records_its_tokenizer_requirements(roster):
    assert "sudachidict_core>=20211220" in roster.pipelines["ja"].extra_requirements


def test_configuration_id_is_deterministic_and_sensitive_to_its_inputs(roster):
    pipeline = roster.pipelines["de"]
    base = configuration_id(pipeline)

    assert configuration_id(load_roster().pipelines["de"]) == base
    assert configuration_id(dataclasses.replace(pipeline, version="3.7.0")) != base
    assert (
        configuration_id(dataclasses.replace(pipeline, package="fr_core_news_sm"))
        != base
    )
    changed_map = dict(pipeline.place_label_map, MISC="LOC")
    assert (
        configuration_id(dataclasses.replace(pipeline, place_label_map=changed_map))
        != base
    )
    assert configuration_id(pipeline, extra={"role": "control"}) != base


def test_configuration_id_is_a_short_hex_string(roster):
    identifier = configuration_id(roster.pipelines["de"])

    assert len(identifier) == 16
    assert int(identifier, 16) >= 0
