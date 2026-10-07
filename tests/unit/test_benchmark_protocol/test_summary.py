"""Hand-derived ratios and bootstrap extremes, independent of implementation."""

import pytest

from scripts.benchmark_protocol.schema import ResolutionCounts
from scripts.benchmark_protocol.summary import aggregate, bootstrap, metrics
from tests.unit.test_benchmark_protocol.test_protocol import completed_payload, parse


def test_count_metrics_match_independent_fractions():
    counts = parse(completed_payload()).results[0].scores
    assert metrics(counts) == {"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3}


def test_resolution_keeps_abstentions_and_missing_gold_in_denominators():
    counts = ResolutionCounts(
        task="gold_span_resolution",
        gold_spans=10,
        resolved=6,
        abstained=3,
        invalid_outputs=1,
        exact_id_eligible=8,
        exact_id_correct=4,
        coordinate_eligible=9,
        within_1km=2,
        within_10km=3,
        within_50km=5,
        candidate_eligible=8,
        candidate_found=7,
    )
    assert metrics(counts) == {
        "exact_id_accuracy": 0.5,
        "accuracy_at_1km": 2 / 9,
        "accuracy_at_10km": 1 / 3,
        "accuracy_at_50km": 5 / 9,
        "coverage": 0.6,
        "abstention_rate": 0.3,
        "invalid_output_rate": 0.1,
        "candidate_recall": 7 / 8,
    }


def test_bootstrap_document_units_has_known_extreme_interval():
    run = parse(completed_payload())
    interval = bootstrap(
        [unit.scores for unit in run.results[0].units], seed=42, resamples=1000
    )
    assert interval == {
        "precision": [0.0, 2 / 3],
        "recall": [0.0, 1.0],
        "f1": [0.0, 0.8],
    }


def test_aggregate_reports_partial_inventory_without_silent_zeros():
    run = parse(completed_payload())
    report = aggregate(run)
    model = report["pipelines"]["fixture"]["0"]
    assert model["languages"] == ["en"]
    assert model["macro"] == {"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3}
    assert model["micro"] == model["macro"]
    assert model["complete"] is False
    assert report["status_counts"] == {"complete": 1, "planned": 1}
    assert model["per_language"]["en"]["counts"]["gold_spans"] == 3
    assert model["uncertainty"]["method"] == "paired_document_bootstrap_95_percent"


@pytest.mark.parametrize(
    "changes",
    [
        {"resolved": 7},
        {"within_1km": 4},
        {"exact_id_correct": 9},
        {"candidate_found": 9},
        {"coordinate_eligible": 11},
    ],
)
def test_resolution_rejects_inconsistent_denominators(changes):
    payload = {
        "task": "gold_span_resolution",
        "gold_spans": 10,
        "resolved": 6,
        "abstained": 3,
        "invalid_outputs": 1,
        "exact_id_eligible": 8,
        "exact_id_correct": 4,
        "coordinate_eligible": 9,
        "within_1km": 2,
        "within_10km": 3,
        "within_50km": 5,
        "candidate_eligible": 8,
        "candidate_found": 7,
    }
    payload.update(changes)
    with pytest.raises(ValueError):
        ResolutionCounts.model_validate(payload)


def test_macro_weights_languages_equally_and_micro_pools_counts():
    payload = completed_payload()
    run = parse(payload)
    empty = {
        "task": "recognition",
        "gold_spans": 9,
        "predicted_spans": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 9,
        "invalid_outputs": 0,
    }
    units = [
        {
            "example_id": str(index),
            "failed": False,
            "scores": dict(empty, gold_spans=gold, false_negative=gold),
        }
        for index, gold in enumerate([2, 1, 1, 1, 1, 1, 1, 1])
    ]
    reference = payload["results"][0]
    payload["results"][1].update(
        status="complete",
        provenance_sha256=run.provenance_digest(run.configurations[1]),
        scores=empty,
        units=units,
        evaluated_examples=8,
        failed_examples=0,
        measurements=reference["measurements"],
        raw_predictions=reference["raw_predictions"],
    )
    report = aggregate(parse(payload))["pipelines"]["fixture"]["0"]
    assert report["complete"] is True
    assert report["macro"] == {"precision": 1 / 3, "recall": 1 / 3, "f1": 1 / 3}
    assert report["micro"] == {"precision": 2 / 3, "recall": 1 / 6, "f1": 4 / 15}
    assert report["per_language"]["fr"]["counts"]["gold_spans"] == 9


def test_unevaluated_inventory_has_no_fabricated_metrics():
    from tests.unit.test_benchmark_protocol.test_protocol import experiment_payload

    assert aggregate(parse(experiment_payload()))["pipelines"] == {
        "fixture": {"0": {"complete": False, "languages": []}}
    }


def test_paired_intervals_ignore_configuration_and_unit_order():
    payload = completed_payload()
    first = aggregate(parse(payload))["pipelines"]
    payload["configurations"].reverse()
    payload["results"].reverse()
    payload["results"][1]["units"].reverse()
    assert aggregate(parse(payload))["pipelines"] == first


@pytest.mark.parametrize("units,resamples", [([], 10), ([], 0)])
def test_bootstrap_rejects_missing_units_or_resamples(units, resamples):
    with pytest.raises(ValueError):
        bootstrap(units, seed=0, resamples=resamples)
