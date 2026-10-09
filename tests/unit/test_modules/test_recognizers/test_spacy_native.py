"""Native-language spaCy recognizer: label harmonization, spans and identity.

Expected offsets are counted by hand from the literal text, so they do not
depend on the recognizer's own arithmetic. No model is downloaded: the pipeline
is a blank English tokenizer with an entity ruler.
"""

import pytest
import spacy

from geoparser.modules.recognizers.spacy_native import NativeSpacyRecognizer

PLACE_MAP = {"GPE": "LOC", "LOC": "LOC"}
RULES = [
    {"label": "GPE", "pattern": "Paris"},
    {"label": "PERSON", "pattern": "Alice"},
    {"label": "LOC", "pattern": "Rhine River"},
]
TEXT = "Alice visited Paris near the Rhine River."


def _pipeline() -> spacy.language.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("entity_ruler").add_patterns(RULES)
    return nlp


def _recognizer(**overrides) -> NativeSpacyRecognizer:
    options = {
        "language": "de",
        "package": "de_core_news_sm",
        "version": "3.8.0",
        "place_label_map": PLACE_MAP,
    }
    options.update(overrides)
    return NativeSpacyRecognizer(_pipeline(), **options)


def test_predict_keeps_only_mapped_place_labels_at_character_offsets():
    # Paris is 14-19 and "Rhine River" is 29-40; the PERSON span is dropped.
    assert _recognizer().predict([TEXT]) == [[(14, 19), (29, 40)]]


def test_predict_batch_returns_sets_aligned_with_inputs():
    assert _recognizer().predict_batch(["Paris", "Alice", ""]) == [
        {(0, 5)},
        set(),
        set(),
    ]


def test_labels_outside_the_harmonized_map_are_never_emitted():
    assert _recognizer(place_label_map={"FAC": "LOC"}).predict_batch(["Paris"]) == [
        set()
    ]


def test_place_label_values_must_be_the_wikiann_location_class():
    with pytest.raises(ValueError, match="LOC"):
        _recognizer(place_label_map={"GPE": "CITY"})


def test_empty_place_label_map_is_rejected():
    with pytest.raises(ValueError, match="place_label_map"):
        _recognizer(place_label_map={})


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_batch_size_must_be_a_positive_integer(batch_size):
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        _recognizer(batch_size=batch_size)


def test_non_string_documents_are_rejected():
    with pytest.raises(TypeError, match="string"):
        _recognizer().predict([TEXT, 3])


def test_configuration_records_language_package_version_and_mapping():
    recognizer = _recognizer()

    assert recognizer.config == {
        "batch_size": 8,
        "language": "de",
        "package": "de_core_news_sm",
        "place_label_map": {"GPE": "LOC", "LOC": "LOC"},
        "version": "3.8.0",
    }


def test_identity_is_deterministic_for_equal_configuration():
    assert _recognizer().id == _recognizer().id


def test_identity_changes_with_language_package_or_mapping():
    base = _recognizer().id

    assert _recognizer(language="fr").id != base
    assert _recognizer(package="fr_core_news_sm").id != base
    assert _recognizer(place_label_map={"GPE": "LOC"}).id != base
