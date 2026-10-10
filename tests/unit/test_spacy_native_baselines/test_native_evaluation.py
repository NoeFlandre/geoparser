"""Matched-language and English-transfer scores stay separate.

Predictors are fakes keyed by sentence text, so the expected counts are
worked out by hand from the gold and predicted spans below.
"""

from types import SimpleNamespace

import pytest

from scripts.panx_benchmark.data import Example
from scripts.spacy_native_baselines import evaluation
from scripts.spacy_native_baselines.evaluation import evaluate_baselines
from scripts.spacy_native_baselines.roster import load_roster


class FakeClock:
    """A perf_counter stand-in that only moves when a predictor advances it."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


class TickingRecognizer:
    """Advance the fake clock on every predict_batch call.

    The first call advances by ``first_step`` (defaulting to ``step``), as a
    model's first inference does while it starts up.
    """

    def __init__(self, clock, step, first_step=None):
        self.clock = clock
        self.step = step
        self.first_step = step if first_step is None else first_step
        self.calls = 0

    def predict_batch(self, texts):
        self.calls += 1
        self.clock.now += self.first_step if self.calls == 1 else self.step
        return [set() for _ in texts]


class FakeRecognizer:
    """Return fixed spans per sentence and record every batch it receives."""

    def __init__(self, spans_by_text):
        self.spans_by_text = spans_by_text
        self.seen = []

    def predict_batch(self, texts):
        self.seen.append(list(texts))
        return [set(self.spans_by_text.get(text, ())) for text in texts]


def _example(language, text, gold):
    return Example(language, text, frozenset(gold))


def _unlabelled(language, count):
    return tuple(
        _example(language, f"sentence {index}", set()) for index in range(count)
    )


def _identify(predictor, roster, language):
    """Record the identity a NativeSpacyRecognizer routed to ``language`` records."""
    pipeline = roster.route(language).pipeline
    predictor.config = {
        "language": pipeline.language,
        "package": pipeline.package,
        "version": pipeline.version,
        "place_label_map": dict(pipeline.place_label_map),
    }
    return predictor


@pytest.fixture
def roster():
    return load_roster()


@pytest.fixture
def examples():
    return {
        "de": (
            _example("de", "Berlin liegt.", {(0, 6)}),
            _example("de", "Nichts hier.", set()),
        ),
        "en": (_example("en", "Paris calls.", {(0, 5)}),),
        "ar": (_example("ar", "مرحبا Paris", {(6, 11)}),),
    }


def test_matched_scores_come_from_the_native_recognizer_only(roster, examples):
    german = _identify(FakeRecognizer({"Berlin liegt.": {(0, 6)}}), roster, "de")
    english = _identify(FakeRecognizer({}), roster, "en")
    control = _identify(FakeRecognizer({"Paris calls.": {(0, 5)}}), roster, "en")

    result = evaluate_baselines(
        examples,
        {"de": german, "en": english},
        control,
        roster,
    )

    assert result["matched"]["de"]["true_positive"] == 1
    assert result["matched"]["de"]["sentences"] == 2
    assert result["matched"]["en"]["true_positive"] == 0
    assert "ar" not in result["matched"]


def test_native_recognizer_receives_only_its_own_language(roster, examples):
    german = _identify(FakeRecognizer({}), roster, "de")

    evaluate_baselines(examples, {"de": german}, None, roster)

    # The untimed warm-up is the first call, then the scored batch.
    assert german.seen == [["Berlin liegt.", "Nichts hier."]] * 2


def test_english_control_transfer_excludes_english_and_unsupported(roster, examples):
    control = _identify(FakeRecognizer({"Berlin liegt.": {(0, 6)}}), roster, "en")
    german = _identify(FakeRecognizer({}), roster, "de")

    result = evaluate_baselines(examples, {"de": german}, control, roster)

    assert set(result["transfer"]) == {"de"}
    assert result["transfer"]["de"]["true_positive"] == 1
    assert result["transfer"]["de"]["false_negative"] == 0


def test_transfer_covers_only_languages_with_a_matched_recognizer(roster):
    examples = {
        "de": (_example("de", "Berlin liegt.", {(0, 6)}),),
        "fr": (_example("fr", "Paris appelle.", {(0, 5)}),),
    }
    control = _identify(FakeRecognizer({}), roster, "en")

    partial = evaluate_baselines(
        examples, {"de": _identify(FakeRecognizer({}), roster, "de")}, control, roster
    )
    full = evaluate_baselines(
        examples,
        {
            "de": _identify(FakeRecognizer({}), roster, "de"),
            "fr": _identify(FakeRecognizer({}), roster, "fr"),
        },
        control,
        roster,
    )

    assert set(partial["matched"]) == set(partial["transfer"]) == {"de"}
    assert set(full["matched"]) == set(full["transfer"]) == {"de", "fr"}


def test_unsupported_languages_are_recorded_and_never_predicted(roster, examples):
    control = _identify(FakeRecognizer({}), roster, "en")

    result = evaluate_baselines(examples, {}, control, roster)

    assert result["unsupported"] == {
        "ar": {"sentences": 1, "reason": roster.route("ar").reason}
    }
    assert "ar" not in result["matched"]
    assert "ar" not in result["transfer"]
    assert all("مرحبا Paris" not in batch for batch in control.seen)


def test_a_run_with_only_unsupported_languages_never_calls_a_model(roster):
    arabic = {"ar": (_example("ar", "مرحبا Paris", {(6, 11)}),)}
    control = _identify(FakeRecognizer({}), roster, "en")

    result = evaluate_baselines(arabic, {}, control, roster)

    assert control.seen == []
    assert result["matched"] == {}
    assert result["transfer"] == {}
    assert result["unsupported"]["ar"]["sentences"] == 1


class ShortRecognizer:
    """Return fewer span sets than the number of input texts."""

    def predict_batch(self, texts):
        return []


def test_a_predictor_must_return_one_span_set_per_input_text(roster):
    examples = {"de": (_example("de", "Berlin liegt.", {(0, 6)}),)}
    short = _identify(ShortRecognizer(), roster, "de")

    with pytest.raises(ValueError, match="one span set per input text"):
        evaluate_baselines(examples, {"de": short}, None, roster)


def test_without_a_control_no_transfer_scores_are_reported(roster, examples):
    german = _identify(FakeRecognizer({}), roster, "de")

    result = evaluate_baselines(examples, {"de": german}, None, roster)

    assert result["transfer"] == {}
    assert result["transfer_macro"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_recognizer_for_an_unsupported_language_is_refused(roster, examples):
    with pytest.raises(ValueError, match="unsupported"):
        evaluate_baselines(examples, {"ar": FakeRecognizer({})}, None, roster)


def test_a_recognizer_must_record_the_pipeline_routed_to_its_language(roster, examples):
    french_under_german = _identify(FakeRecognizer({}), roster, "fr")

    with pytest.raises(ValueError, match="does not record its routed pipeline"):
        evaluate_baselines(examples, {"de": french_under_german}, None, roster)


def test_a_recognizer_without_a_recorded_identity_is_refused(roster, examples):
    with pytest.raises(ValueError, match="does not record its routed pipeline"):
        evaluate_baselines(examples, {"de": FakeRecognizer({})}, None, roster)


def test_the_english_control_must_record_the_english_pipeline(roster, examples):
    german_as_control = _identify(FakeRecognizer({}), roster, "de")

    with pytest.raises(ValueError, match="does not record its routed pipeline"):
        evaluate_baselines(examples, {}, german_as_control, roster)


def test_batches_are_bounded_by_batch_size(roster):
    texts = [f"sentence {index}" for index in range(5)]
    examples = {"de": tuple(_example("de", text, set()) for text in texts)}
    german = _identify(FakeRecognizer({}), roster, "de")

    evaluate_baselines(examples, {"de": german}, None, roster, batch_size=2)

    # Warm-up on the first batch, then the three scored batches.
    assert [len(batch) for batch in german.seen] == [2, 2, 2, 1]


@pytest.mark.parametrize("batch_size", [0, -1, True, 2.0])
def test_batch_size_must_be_a_positive_integer(roster, batch_size):
    examples = {"de": (_example("de", "Berlin liegt.", {(0, 6)}),)}

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        evaluate_baselines(
            examples, {"de": FakeRecognizer({})}, None, roster, batch_size=batch_size
        )


def test_an_example_must_sit_under_its_own_language_before_any_arm_runs(roster):
    misfiled = {"de": (_example("en", "Berlin liegt.", {(0, 6)}),)}
    german = _identify(FakeRecognizer({}), roster, "de")
    control = _identify(FakeRecognizer({}), roster, "en")

    with pytest.raises(ValueError, match="filed under 'de'"):
        evaluate_baselines(misfiled, {"de": german}, control, roster)

    assert german.seen == []
    assert control.seen == []


def test_predictor_time_is_recorded_and_drives_sentences_per_second(
    roster, monkeypatch
):
    clock = FakeClock()
    monkeypatch.setattr(
        evaluation, "time", SimpleNamespace(perf_counter=clock.perf_counter)
    )
    examples = {"de": _unlabelled("de", 3)}
    german = _identify(TickingRecognizer(clock, step=2.0), roster, "de")
    control = _identify(TickingRecognizer(clock, step=1.0), roster, "en")

    result = evaluate_baselines(
        examples,
        {"de": german},
        control,
        roster,
        batch_size=2,
    )

    matched = result["matched"]["de"]
    assert matched["elapsed_seconds"] == pytest.approx(4.0)
    assert matched["sentences_per_second"] == pytest.approx(3 / 4.0)
    transfer = result["transfer"]["de"]
    assert transfer["elapsed_seconds"] == pytest.approx(2.0)
    assert transfer["sentences_per_second"] == pytest.approx(3 / 2.0)


def test_warm_up_runs_before_the_timers_and_is_not_timed(roster, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(
        evaluation, "time", SimpleNamespace(perf_counter=clock.perf_counter)
    )
    examples = {"de": _unlabelled("de", 3)}
    # A slow first call stands in for model startup; it must not be timed.
    german = _identify(
        TickingRecognizer(clock, step=2.0, first_step=100.0), roster, "de"
    )

    result = evaluate_baselines(examples, {"de": german}, None, roster, batch_size=2)

    assert german.calls == 3  # warm-up, then the two scored batches
    assert result["matched"]["de"]["elapsed_seconds"] == pytest.approx(4.0)


def test_malformed_gold_tags_are_counted_for_matched_and_transfer_scores(roster):
    malformed = Example(
        "de", "Berlin liegt.", frozenset({(0, 6)}), malformed_location_tags=2
    )

    result = evaluate_baselines(
        {"de": (malformed,)},
        {"de": _identify(FakeRecognizer({}), roster, "de")},
        _identify(FakeRecognizer({"Berlin liegt.": {(0, 6)}}), roster, "en"),
        roster,
    )

    assert result["matched"]["de"]["malformed_gold_tags"] == 2
    assert result["transfer"]["de"]["malformed_gold_tags"] == 2


def test_macro_scores_average_languages_with_equal_weight(roster, examples):
    german = _identify(FakeRecognizer({"Berlin liegt.": {(0, 6)}}), roster, "de")

    result = evaluate_baselines(
        examples,
        {"de": german},
        None,
        roster,
    )

    assert result["matched_macro"]["precision"] == pytest.approx(1.0)
    assert result["matched_macro"]["recall"] == pytest.approx(1.0)


def test_paired_matched_macro_drops_the_native_english_score(roster):
    examples = {
        "en": (_example("en", "Paris calls.", {(0, 5)}),),
        "de": (_example("de", "Berlin liegt.", {(0, 6)}),),
    }
    english = _identify(FakeRecognizer({"Paris calls.": {(0, 5)}}), roster, "en")
    german = _identify(FakeRecognizer({}), roster, "de")
    control = _identify(FakeRecognizer({"Berlin liegt.": {(0, 6)}}), roster, "en")

    result = evaluate_baselines(
        examples, {"de": german, "en": english}, control, roster
    )

    assert (set(result["matched"]), set(result["transfer"])) == (
        {"de", "en"},
        {"de"},
    )
    assert result["matched_macro"]["recall"] == pytest.approx(0.5)
    assert result["paired_matched_macro"]["recall"] == pytest.approx(0.0)
    assert result["transfer_macro"]["recall"] == pytest.approx(1.0)
