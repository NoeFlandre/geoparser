"""Native-language spaCy recognizer: label harmonization, spans and identity.

Expected offsets are counted by hand from the literal text, so they do not
depend on the recognizer's own arithmetic. No model is downloaded: the pipeline
is a blank English tokenizer with an entity ruler.
"""

from typing import Any, cast

import pytest
import spacy
from spacy.pipeline import EntityRuler

from geoparser.modules.recognizers.spacy_native import NativeSpacyRecognizer

PLACE_MAP = {"GPE": "LOC", "LOC": "LOC"}
RULES: list[dict[str, Any]] = [
    {"label": "GPE", "pattern": "Paris"},
    {"label": "PERSON", "pattern": "Alice"},
    {"label": "LOC", "pattern": "Rhine River"},
]
TEXT = "Alice visited Paris near the Rhine River."
OPTIONS: dict[str, Any] = {
    "language": "de",
    "package": "de_core_news_sm",
    "version": "3.8.0",
    "place_label_map": PLACE_MAP,
}


class RecordingPipeline:
    """Delegate to a real pipeline and record the batch size each call requests."""

    def __init__(self, nlp: spacy.language.Language) -> None:
        self.nlp = nlp
        self.requested_batch_sizes: list[Any] = []

    def pipe(self, texts: list[str], *, batch_size: Any) -> Any:
        self.requested_batch_sizes.append(batch_size)
        return self.nlp.pipe(texts, batch_size=batch_size)


def _pipeline() -> spacy.language.Language:
    nlp = spacy.blank("en")
    # spaCy annotates add_pipe as returning the pipeline callable, not the ruler.
    ruler = cast(EntityRuler, nlp.add_pipe("entity_ruler"))
    ruler.add_patterns(RULES)
    return nlp


def _recognizer(nlp: Any = None, **overrides: Any) -> NativeSpacyRecognizer:
    options = {**OPTIONS, **overrides}
    return NativeSpacyRecognizer(nlp if nlp is not None else _pipeline(), **options)


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
    with pytest.raises(
        ValueError, match=r"^place_label_map must be a nonempty mapping\.$"
    ):
        _recognizer(place_label_map={})


@pytest.mark.parametrize("label", ["", "   "])
def test_blank_place_labels_are_rejected(label):
    with pytest.raises(
        ValueError, match=r"^place_label_map label must be a nonblank string\.$"
    ):
        _recognizer(place_label_map={label: "LOC"})


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_batch_size_must_be_a_positive_integer(batch_size):
    with pytest.raises(ValueError, match=r"^batch_size must be a positive integer\.$"):
        _recognizer(batch_size=batch_size)


def test_batch_size_one_is_the_smallest_accepted_value():
    recognizer = _recognizer(batch_size=1)

    assert recognizer.config["batch_size"] == 1
    assert recognizer.predict_batch([TEXT]) == [{(14, 19), (29, 40)}]


def test_predict_batch_passes_the_configured_batch_size_to_the_pipeline():
    pipeline = RecordingPipeline(_pipeline())
    recognizer = _recognizer(pipeline, batch_size=2)

    assert recognizer.predict_batch([TEXT, "Paris"]) == [
        {(14, 19), (29, 40)},
        {(0, 5)},
    ]
    assert pipeline.requested_batch_sizes == [2]


@pytest.mark.parametrize("field", ["language", "package", "version"])
@pytest.mark.parametrize("value", ["", "   "])
def test_identity_fields_must_be_nonblank(field, value):
    with pytest.raises(ValueError, match=rf"^{field} must be a nonblank string\.$"):
        _recognizer(**{field: value})


@pytest.mark.parametrize("field", ["language", "package", "version"])
def test_identity_fields_must_be_strings(field):
    with pytest.raises(ValueError, match=rf"^{field} must be a nonblank string\.$"):
        _recognizer(**{field: 3})


@pytest.mark.parametrize("texts", ["Paris", ("Paris",)])
def test_documents_must_be_passed_as_a_list(texts):
    with pytest.raises(TypeError, match=r"^texts must be a list of strings\.$"):
        _recognizer().predict_batch(texts)


def test_non_string_documents_are_rejected():
    mixed: list[Any] = [TEXT, 3]

    with pytest.raises(
        TypeError, match=r"^Every document in texts must be a string\.$"
    ):
        _recognizer().predict(mixed)


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
