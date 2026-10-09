"""Matched-language and English-transfer scores stay separate.

Predictors are fakes keyed by sentence text, so the expected counts are
worked out by hand from the gold and predicted spans below.
"""

import pytest

from scripts.panx_benchmark.data import Example
from scripts.spacy_native_baselines.evaluation import evaluate_baselines
from scripts.spacy_native_baselines.roster import load_roster


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
    german = FakeRecognizer({"Berlin liegt.": {(0, 6)}})
    english = FakeRecognizer({})
    control = FakeRecognizer({"Paris calls.": {(0, 5)}})

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
    assert german.seen == [["Berlin liegt.", "Nichts hier."]]


def test_english_control_transfer_is_separate_and_excludes_english(roster, examples):
    control = FakeRecognizer({"Berlin liegt.": {(0, 6)}})

    result = evaluate_baselines(examples, {"de": FakeRecognizer({})}, control, roster)

    assert set(result["transfer"]) == {"de", "ar"}
    assert "en" not in result["transfer"]
    assert result["transfer"]["de"]["true_positive"] == 1
    assert result["transfer"]["de"]["false_negative"] == 0


def test_unsupported_languages_are_recorded_and_never_predicted(roster, examples):
    control = FakeRecognizer({})

    result = evaluate_baselines(examples, {}, control, roster)

    assert result["unsupported"] == {
        "ar": {"sentences": 1, "reason": roster.route("ar").reason}
    }
    assert "ar" not in result["matched"]
    # The English control may score Arabic only inside the labelled transfer group.
    assert ["مرحبا Paris"] in control.seen
    assert result["transfer"]["ar"]["false_negative"] == 1


def test_without_a_control_no_transfer_scores_are_reported(roster, examples):
    result = evaluate_baselines(examples, {"de": FakeRecognizer({})}, None, roster)

    assert result["transfer"] == {}
    assert result["transfer_macro"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_recognizer_for_an_unsupported_language_is_refused(roster, examples):
    with pytest.raises(ValueError, match="unsupported"):
        evaluate_baselines(examples, {"ar": FakeRecognizer({})}, None, roster)


def test_batches_are_bounded_by_batch_size(roster):
    texts = [f"sentence {index}" for index in range(5)]
    examples = {"de": tuple(_example("de", text, set()) for text in texts)}
    german = FakeRecognizer({})

    evaluate_baselines(examples, {"de": german}, None, roster, batch_size=2)

    assert [len(batch) for batch in german.seen] == [2, 2, 1]


def test_macro_scores_average_languages_with_equal_weight(roster, examples):
    german = FakeRecognizer({"Berlin liegt.": {(0, 6)}})

    result = evaluate_baselines(
        examples,
        {"de": german},
        None,
        roster,
    )

    assert result["matched_macro"]["precision"] == pytest.approx(1.0)
    assert result["matched_macro"]["recall"] == pytest.approx(1.0)
