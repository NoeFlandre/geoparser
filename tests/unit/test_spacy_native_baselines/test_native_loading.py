"""Pinned native pipeline loading: missing packages fail without downloads."""

from typing import cast

import pytest
import spacy
from spacy.pipeline import EntityRecognizer

from scripts.spacy_native_baselines.loading import (
    LabelSchemeError,
    MissingPipelineError,
    SpacyRuntimeError,
    check_installed,
    check_spacy_runtime,
    installed_version,
    load_pipeline,
    requirement_name,
    resolved_versions,
    run_identity,
)
from scripts.spacy_native_baselines.roster import load_roster


@pytest.fixture
def german():
    return load_roster().pipelines["de"]


@pytest.fixture
def no_download(monkeypatch):
    def refuse(*args, **kwargs):
        message = "the baseline must never download a pipeline"
        raise AssertionError(message)

    monkeypatch.setattr(spacy.cli, "download", refuse)


def _add_ner(nlp):
    # spaCy annotates add_pipe as returning the pipeline callable, not the NER.
    return cast(EntityRecognizer, nlp.add_pipe("ner"))


def _blank_with_ner(labels):
    nlp = spacy.blank("de")
    ner = _add_ner(nlp)
    for label in labels:
        ner.add_label(label)
    return nlp


def test_missing_package_names_the_pinned_release_and_never_downloads(
    german, no_download
):
    with pytest.raises(MissingPipelineError) as raised:
        check_installed(german, version_lookup=lambda package: None)

    message = str(raised.value)
    assert "de_core_news_sm" in message
    assert german.wheel_url in message
    assert "not" in message.lower()


def test_missing_japanese_pipeline_guidance_names_its_tokenizer_requirements(
    no_download,
):
    japanese = load_roster().pipelines["ja"]

    with pytest.raises(MissingPipelineError) as raised:
        check_installed(japanese, version_lookup=lambda package: None)

    message = str(raised.value)
    assert japanese.wheel_url in message
    assert "'sudachidict_core>=20211220'" in message


def test_version_mismatch_names_both_installed_and_pinned_versions(german, no_download):
    with pytest.raises(MissingPipelineError) as raised:
        check_installed(german, version_lookup=lambda package: "3.7.0")

    assert "3.7.0" in str(raised.value)
    assert "3.8.0" in str(raised.value)


def test_matching_installed_version_passes(german, no_download):
    check_installed(german, version_lookup=lambda package: "3.8.0")


def test_installed_version_reads_the_distribution_metadata_or_none():
    assert installed_version("spacy") == spacy.__version__
    assert installed_version("geoparser-no-such-distribution") is None


@pytest.mark.parametrize("found", ["3.8.0", "3.8.16", "3.8.0.post1", "3.8.1rc1"])
def test_spacy_runtime_inside_the_roster_range_passes(found, no_download):
    check_spacy_runtime(version_lookup=lambda package: found)


@pytest.mark.parametrize(
    "found",
    ["3.7.1", "3.8.0rc1", "3.8.0.dev1", "3.9.0", "3.9.0rc1", "4.0.0", "not-a-version"],
)
def test_spacy_runtime_outside_the_roster_range_is_refused(found, no_download):
    with pytest.raises(SpacyRuntimeError) as raised:
        check_spacy_runtime(version_lookup=lambda package: found)

    assert found in str(raised.value)
    assert ">=3.8.0,<3.9.0" in str(raised.value)


def test_missing_spacy_runtime_is_refused(no_download):
    with pytest.raises(SpacyRuntimeError, match="not installed"):
        check_spacy_runtime(version_lookup=lambda package: None)


def test_load_pipeline_refuses_a_spacy_runtime_outside_the_pin(german, no_download):
    versions = {"de_core_news_sm": "3.8.0", "spacy": "3.9.0"}

    def loader(name):
        message = "loader must not run for an out-of-range spaCy runtime"
        raise AssertionError(message)

    with pytest.raises(SpacyRuntimeError, match=r"spaCy 3\.9\.0 is installed"):
        load_pipeline(german, loader=loader, version_lookup=versions.get)


def test_loader_is_not_called_when_the_package_is_missing(german, no_download):
    def loader(name):
        message = "loader must not run for a missing package"
        raise AssertionError(message)

    with pytest.raises(MissingPipelineError):
        load_pipeline(german, loader=loader, version_lookup=lambda package: None)


def test_loaded_pipeline_must_expose_exactly_the_roster_ner_labels(german):
    with pytest.raises(LabelSchemeError, match="MISC"):
        load_pipeline(
            german,
            loader=lambda name: _blank_with_ner(["LOC", "PER"]),
            version_lookup=lambda package: "3.8.0",
        )


def test_loaded_pipeline_keeps_only_ner_and_its_tok2vec_listener(german):
    labels = list(german.ner_labels)
    nlp = spacy.blank("de")
    nlp.add_pipe("tagger")
    ner = _add_ner(nlp)
    for label in labels:
        ner.add_label(label)

    loaded = load_pipeline(
        german,
        loader=lambda name: nlp,
        version_lookup=lambda package: "3.8.0",
    )

    assert "tagger" not in loaded.pipe_names
    assert "ner" in loaded.pipe_names


@pytest.mark.parametrize(
    ("requirement", "name"),
    [
        ("sudachipy!=0.6.1,>=0.5.2", "sudachipy"),
        ("sudachidict_core>=20211220", "sudachidict_core"),
        ("natto-py>=0.9.0", "natto-py"),
        ("spacy", "spacy"),
    ],
)
def test_requirement_name_is_the_distribution_before_its_specifier(requirement, name):
    assert requirement_name(requirement) == name


def test_requirement_without_a_distribution_name_is_refused():
    with pytest.raises(ValueError, match="distribution name"):
        requirement_name("!>=1.0")


def test_resolved_versions_name_spacy_the_pipeline_and_each_tokenizer():
    japanese = load_roster().pipelines["ja"]

    versions = resolved_versions(japanese, version_lookup=lambda name: "1.0")

    assert set(versions) == {
        "spacy",
        "ja_core_news_sm",
        "sudachipy",
        "sudachidict_core",
    }


def test_run_identity_changes_with_the_installed_tokenizer_release():
    japanese = load_roster().pipelines["ja"]
    installed = {
        "spacy": "3.8.16",
        "ja_core_news_sm": "3.8.0",
        "sudachipy": "0.6.9",
        "sudachidict_core": "20250129",
    }

    base = run_identity(japanese, version_lookup=installed.get)

    assert run_identity(japanese, version_lookup=installed.get) == base
    assert (
        run_identity(japanese, version_lookup=dict(installed, sudachipy="0.7.0").get)
        != base
    )
    assert (
        run_identity(japanese, version_lookup=dict(installed, spacy="3.8.0").get)
        != base
    )
