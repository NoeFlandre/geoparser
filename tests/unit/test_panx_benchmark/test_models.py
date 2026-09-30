from scripts.panx_benchmark.constants import MODELS
from scripts.panx_benchmark.models import (
    GLiNERPredictor,
    SpacyRecognizer,
    XLMRecognizer,
    _gliner_location_spans,
    _xlm_location_spans,
)
from scripts.panx_benchmark.runner import _location_mapping


def test_gliner_maps_all_project_place_types_to_one_deduplicated_loc_set():
    spans = _gliner_location_spans(
        {
            "entities": {
                "city": [{"start": 0, "end": 4}],
                "country": [{"start": 0, "end": 4}],
                "location": [{"start": 9, "end": 14}],
                "person": [{"start": 16, "end": 20}],
            }
        }
    )

    assert spans == {(0, 4), (9, 14)}


def test_xlmr_keeps_only_loc_and_uses_character_offsets():
    spans = _xlm_location_spans(
        [
            {"entity_group": "PER", "start": 0, "end": 4},
            {"entity_group": "LOC", "start": 5, "end": 10},
            {"entity_group": "ORG", "start": 11, "end": 15},
        ]
    )

    assert spans == {(5, 10)}


def test_xlmr_handles_bio_label_fallback():
    assert _xlm_location_spans([{"entity": "B-LOC", "start": 2, "end": 7}]) == {(2, 7)}


def test_gliner_adapter_uses_fixed_labels_batch_size_and_threshold():
    class FakeModel:
        def __init__(self):
            self.calls = []

        def batch_extract_entities(self, texts, labels, **kwargs):
            self.calls.append((texts, labels, kwargs))
            return [{"entities": {"location": [{"start": 0, "end": 4}]}}]

    model = FakeModel()
    predictions = GLiNERPredictor(model).predict_batch(["Town"])

    assert (predictions, model.calls) == (
        [{(0, 4)}],
        [
            (
                ["Town"],
                ["city", "country", "location"],
                {"batch_size": 8, "threshold": 0.5, "include_spans": True},
            )
        ],
    )


def test_xlmr_adapter_calls_pipeline_and_maps_location_spans():
    class FakePipeline:
        def __call__(self, texts, *, batch_size):
            assert texts == ["Town"]
            assert batch_size == 8
            return [[{"entity_group": "LOC", "start": 0, "end": 4}]]

    assert XLMRecognizer(FakePipeline()).predict_batch(["Town"]) == [{(0, 4)}]


def test_spacy_adapter_uses_project_location_labels():
    class Entity:
        start_char = 0
        end_char = 4
        label_ = "GPE"

    class Document:
        def __init__(self):
            self.ents = (Entity(),)

    class Pipeline:
        def pipe(self, texts, *, batch_size):
            assert texts == ["Town"]
            assert batch_size == 8
            return [Document()]

    assert SpacyRecognizer(Pipeline()).predict_batch(["Town"]) == [{(0, 4)}]


def test_spacy_location_mapping_matches_the_existing_recognizer_defaults():
    spacy = next(spec for spec in MODELS if spec.key == "spacy_en")

    assert _location_mapping(spacy) == "FAC, GPE, LOC -> WikiANN LOC"
