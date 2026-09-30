from scripts.panx_benchmark.constants import MODELS
from scripts.panx_benchmark.data import Example, LoadedDataset
from scripts.panx_benchmark.models import LoadedModel
from scripts.panx_benchmark.runner import _micro_scores, evaluate_model


class ExactPredictor:
    def predict_batch(self, texts):
        return [{(0, len(text))} for text in texts]


def _dataset():
    return LoadedDataset(
        examples_by_language={
            "en": (Example("en", "Paris", frozenset({(0, 5)})),),
            "fr": (Example("fr", "Rome", frozenset({(0, 4)})),),
        },
        source_counts={"en": 3, "fr": 4},
        load_seconds=0.25,
        limit_per_language=1,
    )


def _loaded(predictor, *, download_seconds=0.0, load_seconds=1.0, cache_hit=False):
    return LoadedModel(
        predictor,
        download_seconds,
        load_seconds,
        cache_hit,
        "/cache/model",
    )


def test_micro_scores_aggregate_steady_inference_time():
    scores = _micro_scores(
        {
            "en": {
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 0,
                "sentences": 2,
                "elapsed_seconds": 0.25,
            },
            "fr": {
                "true_positive": 2,
                "false_positive": 0,
                "false_negative": 1,
                "sentences": 3,
                "elapsed_seconds": 0.75,
            },
        }
    )

    assert scores["precision"] == 0.75
    assert scores["recall"] == 0.75
    assert scores["elapsed_seconds"] == 1.0
    assert scores["sentences_per_second"] == 5.0


def test_multilingual_evaluation_scores_all_languages_after_warmup():
    result = evaluate_model(
        MODELS[1],
        _loaded(ExactPredictor(), download_seconds=1.0, load_seconds=2.0),
        _dataset(),
    )

    assert result["evaluated_examples"] == 2
    assert result["warmup_examples"] == 1
    assert result["macro"]["f1"] == 1.0
    assert result["per_language"]["fr"]["status"] == "evaluated"
    assert result["per_language"]["fr"]["metrics"]["sentences"] == 1


def test_spacy_is_reported_only_for_english():
    result = evaluate_model(
        MODELS[0],
        _loaded(ExactPredictor()),
        _dataset(),
    )

    assert result["evaluated_examples"] == 1
    assert result["per_language"]["en"]["status"] == "evaluated"
    assert result["per_language"]["fr"]["status"] == "not_evaluated_english_only"
    assert result["macro"]["f1"] == 1.0


def test_evaluator_rejects_spans_outside_the_source_text():
    class InvalidPredictor:
        def predict_batch(self, texts):
            return [{(0, len(text) + 1)} for text in texts]

    try:
        evaluate_model(
            MODELS[1],
            _loaded(InvalidPredictor()),
            _dataset(),
        )
    except ValueError as error:
        assert "invalid span" in str(error)
    else:
        raise AssertionError("An out-of-text model span must fail evaluation")
