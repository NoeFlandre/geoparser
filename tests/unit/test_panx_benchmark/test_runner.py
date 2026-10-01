import pytest

from scripts.panx_benchmark.checkpoint import ModelLanguageCheckpoints
from scripts.panx_benchmark.constants import MODELS
from scripts.panx_benchmark.data import Example, LoadedDataset
from scripts.panx_benchmark.models import LoadedModel
from scripts.panx_benchmark.runner import (
    _elapsed_full_matrix_seconds,
    _language_results,
    _micro_scores,
    _support_status,
    _valid_predictions,
    _warm_up,
    evaluate_model,
)


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

    assert (
        result["evaluated_examples"],
        result["warmup_examples"],
        result["macro"]["f1"],
        result["per_language"]["fr"]["status"],
        result["per_language"]["fr"]["metrics"]["sentences"],
    ) == (2, 1, 1.0, "evaluated", 1)


def test_gliner_group_size_is_used_for_warmup_and_every_inference_call():
    class RecordingPredictor(ExactPredictor):
        def __init__(self):
            self.group_sizes = []

        def predict_batch(self, texts):
            self.group_sizes.append(len(texts))
            return super().predict_batch(texts)

    examples = tuple(Example("en", f"Town {index}", frozenset()) for index in range(3))
    dataset = LoadedDataset(
        examples_by_language={"en": examples},
        source_counts={"en": len(examples)},
        load_seconds=0.0,
        limit_per_language=None,
    )
    predictor = RecordingPredictor()

    result = evaluate_model(MODELS[1], _loaded(predictor), dataset)

    assert (result["batch_size"], result["warmup_examples"]) == (1, 1)
    assert result["evaluated_examples"] == 3
    assert predictor.group_sizes == [1, 1, 1, 1]


def test_completed_language_checkpoints_prevent_duplicate_inference(tmp_path):
    class CountingPredictor(ExactPredictor):
        def __init__(self):
            self.batches = 0

        def predict_batch(self, texts):
            self.batches += 1
            return super().predict_batch(texts)

    predictor = CountingPredictor()
    checkpoints = ModelLanguageCheckpoints(tmp_path, {"repository_commit": "abc123"})

    first = _language_results(MODELS[1], predictor, _dataset(), checkpoints)
    resumed = _language_results(MODELS[1], predictor, _dataset(), checkpoints)

    assert (predictor.batches, resumed) == (2, first)


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


def test_evaluator_counts_spans_outside_the_source_as_false_positives():
    class InvalidPredictor:
        def predict_batch(self, texts):
            return [{(0, len(text) + 1)} for text in texts]

    result = evaluate_model(MODELS[1], _loaded(InvalidPredictor()), _dataset())
    metrics = result["per_language"]["en"]["metrics"]

    assert (
        metrics["false_positive"],
        metrics["false_negative"],
        metrics["invalid_prediction_spans"],
    ) == (1, 1, 1)


def test_support_status_separates_documented_support_from_transfer():
    spacy, gliner, xlmr = MODELS

    assert (
        _support_status(spacy, "en"),
        _support_status(spacy, "fr"),
        _support_status(gliner, "fr"),
        _support_status(xlmr, "ar"),
        _support_status(xlmr, "ha"),
    ) == (
        "documented",
        "not_evaluated_english_only",
        "evaluated_multilingual_claim",
        "fine_tuned_language",
        "cross_lingual_transfer",
    )


def test_warm_up_skips_languages_outside_the_model_support():
    seconds, examples = _warm_up(
        ExactPredictor(),
        {"fr": (Example("fr", "Lyon", frozenset()),)},
        ("en",),
        8,
    )

    assert (seconds, examples) == (0.0, 0)


def test_valid_predictions_reject_a_mismatched_batch_size():
    class EmptyPredictor:
        def predict_batch(self, texts):
            return []

    with pytest.raises(ValueError, match="different number"):
        _valid_predictions(EmptyPredictor(), [Example("en", "Paris", frozenset())])


def test_full_matrix_estimate_requires_every_model_estimate():
    assert (
        _elapsed_full_matrix_seconds([]),
        _elapsed_full_matrix_seconds(
            [
                {"full_matrix_estimated_inference_seconds": 2.5},
                {"full_matrix_estimated_inference_seconds": 1.5},
            ]
        ),
        _elapsed_full_matrix_seconds(
            [{"full_matrix_estimated_inference_seconds": None}]
        ),
    ) == (0, 4.0, None)


def test_scoring_timer_excludes_metric_bookkeeping(monkeypatch):
    from scripts.panx_benchmark import runner

    clock = {"seconds": 0.0}
    original_add = runner.Counts.add

    class TimedPredictor(ExactPredictor):
        def predict_batch(self, texts):
            clock["seconds"] += 2.0
            return super().predict_batch(texts)

    def slow_add(self, *args, **kwargs):
        clock["seconds"] += 100.0
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock["seconds"])
    monkeypatch.setattr(runner.Counts, "add", slow_add)
    examples = (Example("en", "Paris", frozenset({(0, 5)})),) * 3
    counts = runner._score_language(TimedPredictor(), examples, batch_size=2)
    assert counts.elapsed_seconds == 4.0
    assert counts.sentences == 3
    assert clock["seconds"] == 304.0


def test_micro_scores_preserve_malformed_gold_counts():
    from scripts.panx_benchmark.metrics import Counts

    result = _micro_scores(
        {
            "en": Counts(malformed_gold_tags=2).scores(),
            "fr": Counts(malformed_gold_tags=3).scores(),
        }
    )
    assert result["malformed_gold_tags"] == 5
