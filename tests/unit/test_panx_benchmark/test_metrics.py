from scripts.panx_benchmark.metrics import Counts, macro_scores


def test_counts_use_exact_character_spans_and_zero_division():
    counts = Counts()
    counts.add({(0, 4), (5, 9)}, {(0, 4), (10, 14)})

    assert counts.scores() == {
        "sentences": 1,
        "gold_spans": 2,
        "predicted_spans": 2,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "malformed_gold_tags": 0,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "elapsed_seconds": 0.0,
        "sentences_per_second": 0.0,
    }


def test_empty_gold_and_predictions_are_zero_not_undefined():
    scores = Counts().scores()

    assert scores["precision"] == scores["recall"] == scores["f1"] == 0


def test_macro_scores_weight_languages_equally():
    macro = macro_scores(
        {
            "a": {"precision": 1.0, "recall": 0.0, "f1": 0.0},
            "b": {"precision": 0.0, "recall": 1.0, "f1": 0.0},
        }
    )

    assert macro == {"precision": 0.5, "recall": 0.5, "f1": 0.0}
