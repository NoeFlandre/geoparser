"""Independent, hand-counted oracles for the public comparison contract."""

import copy
import json

import pytest
from pydantic import ValidationError

from scripts.benchmark_protocol.schema import Experiment


def experiment_payload():
    """A two-language inventory with deliberately unequal denominators."""
    artifact = {
        "identifier": "synthetic-public-fixture",
        "revision": "a" * 40,
        "sha256": "b" * 64,
    }
    protocol = {
        "schema_version": "1.0",
        "task": "recognition",
        "code_revision": "c" * 40,
        "dependency_lock_sha256": "d" * 64,
        "dataset": artifact,
        "annotation_quality": "human_gold",
        "split": "development",
        "stage": "screening",
        "threshold_selection": "development",
        "selection_sha256": None,
        "hardware": {
            "platform": "fixture-linux",
            "processor": "fixture-cpu",
            "device": "cpu",
            "threads": 1,
            "runtime_versions": {"python": "3.12"},
        },
        "timing_policy": "separate_fetch_load_warmup_steady",
        "memory_policy": "process_peak_rss_and_device_peak_bytes",
        "invalid_output_policy": "count_as_false_positive_and_retain",
        "uncertainty": "paired_document_bootstrap_95_percent",
        "bootstrap_seed": 42,
        "bootstrap_resamples": 1000,
    }
    configs = []
    for language, examples, gold in [("en", 2, 3), ("fr", 8, 9)]:
        configs.append(
            {
                "key": language,
                "pipeline": "fixture",
                "language": language,
                "source_config": language,
                "examples": examples,
                "gold_spans": gold,
                "sample_sha256": "e" * 64,
                "models": [artifact],
                "label_mapping": {"GPE": "LOC", "LOC": "LOC"},
                "thresholds": {"recognition": 0.5},
                "batch_size": 1,
                "seed": 0,
                "language_support": "documented",
                "training_overlap": "unknown",
                "provenance_note": "Synthetic fixture, no training claim.",
                "gazetteer": None,
            }
        )
    return {
        "protocol": protocol,
        "configurations": configs,
        "results": [
            {
                "key": key,
                "status": "planned",
                "reason": None,
                "provenance_sha256": None,
                "measurements": None,
                "scores": None,
                "raw_predictions": None,
                "evaluated_examples": 0,
                "failed_examples": 0,
            }
            for key in ["en", "fr"]
        ],
    }


def parse(payload):
    """Use the public JSON boundary, including its strict numeric types."""
    return Experiment.model_validate_json(json.dumps(payload))


def test_accepts_explicit_planned_inventory():
    run = parse(experiment_payload())
    assert [item.key for item in run.configurations] == ["en", "fr"]
    assert run.protocol.schema_version == "1.0"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0"),
        ("code_revision", "main"),
        ("dependency_lock_sha256", "unknown"),
        ("split", "train"),
        ("bootstrap_seed", True),
        ("bootstrap_resamples", 0),
    ],
)
def test_rejects_invalid_protocol(field, value):
    payload = experiment_payload()
    payload["protocol"][field] = value
    with pytest.raises(ValidationError):
        parse(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("language", "xx"),
        ("examples", True),
        ("gold_spans", -1),
        ("batch_size", 0),
        ("seed", 1.5),
        ("label_mapping", {}),
        ("thresholds", {"recognition": float("nan")}),
    ],
)
def test_rejects_invalid_configuration(field, value):
    payload = experiment_payload()
    payload["configurations"][0][field] = value
    with pytest.raises(ValidationError):
        parse(payload)


@pytest.mark.parametrize("mutation", ["omitted", "duplicate", "unknown"])
def test_rejects_silently_skipped_or_duplicated_configurations(mutation):
    payload = experiment_payload()
    if mutation == "omitted":
        payload["results"].pop()
    elif mutation == "duplicate":
        payload["results"][1] = copy.deepcopy(payload["results"][0])
    else:
        payload["results"][1]["key"] = "absent"
    with pytest.raises(ValidationError, match="inventory"):
        parse(payload)


@pytest.mark.parametrize(
    "stage,split,selection",
    [
        ("screening", "test", None),
        ("final", "development", "f" * 64),
        ("final", "test", None),
    ],
)
def test_prevents_test_screening_and_unfrozen_finalists(stage, split, selection):
    payload = experiment_payload()
    payload["protocol"].update(stage=stage, split=split, selection_sha256=selection)
    with pytest.raises(ValidationError, match="selection"):
        parse(payload)


def test_accepts_frozen_finalists_with_explicit_fixed_threshold():
    payload = experiment_payload()
    payload["protocol"].update(
        stage="final",
        split="test",
        selection_sha256="f" * 64,
        threshold_selection="fixed_before_evaluation",
    )
    assert parse(payload).protocol.split == "test"


def test_requires_failure_and_unsupported_reasons():
    for status in ["failed", "unsupported"]:
        payload = experiment_payload()
        payload["results"][0]["status"] = status
        with pytest.raises(ValidationError, match="reason"):
            parse(payload)


def test_duplicate_model_cannot_change_source_denominators():
    payload = experiment_payload()
    duplicate = copy.deepcopy(payload["configurations"][0])
    duplicate.update(key="other-model", pipeline="other", gold_spans=2)
    payload["configurations"].append(duplicate)
    outcome = copy.deepcopy(payload["results"][0])
    outcome["key"] = "other-model"
    payload["results"].append(outcome)
    with pytest.raises(ValidationError, match="source"):
        parse(payload)


def completed_payload():
    """Two hand-counted documents: (TP, FP, FN) = (2, 1, 0), (0, 0, 1)."""
    payload = experiment_payload()
    run = parse(payload)
    counts = {
        "task": "recognition",
        "gold_spans": 3,
        "predicted_spans": 3,
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 1,
        "invalid_outputs": 1,
    }
    units = [
        {
            "example_id": "first",
            "failed": False,
            "scores": dict(counts, gold_spans=2, false_negative=0),
        },
        {
            "example_id": "second",
            "failed": True,
            "scores": dict(
                counts,
                gold_spans=1,
                predicted_spans=0,
                true_positive=0,
                false_positive=0,
                false_negative=1,
                invalid_outputs=0,
            ),
        },
    ]
    payload["results"][0].update(
        status="complete",
        provenance_sha256=run.provenance_digest(run.configurations[0]),
        scores=counts,
        units=units,
        evaluated_examples=1,
        failed_examples=1,
        measurements={
            "fetch_seconds": 2.0,
            "load_seconds": 3.0,
            "warmup_seconds": 0.5,
            "steady_seconds": 4.0,
            "warmup_examples": 1,
            "peak_rss_bytes": 1024,
            "peak_device_bytes": 0,
        },
        raw_predictions={
            "identifier": "retained-predictions.jsonl",
            "revision": "1",
            "sha256": "9" * 64,
        },
    )
    return payload


def test_completed_results_bind_provenance_and_account_for_failures():
    run = parse(completed_payload())
    assert run.results[0].scores.true_positive == 2
    assert run.results[0].failed_examples == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("provenance_sha256", "0" * 64),
        ("evaluated_examples", 0),
        ("failed_examples", 3),
        ("raw_predictions", None),
        ("units", []),
    ],
)
def test_rejects_incomplete_or_mixed_completed_evidence(field, value):
    payload = completed_payload()
    payload["results"][0][field] = value
    with pytest.raises(ValidationError):
        parse(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gold_spans", 4),
        ("predicted_spans", 2),
        ("invalid_outputs", 2),
        ("false_positive", -1),
        ("true_positive", True),
    ],
)
def test_rejects_bad_span_count_denominators(field, value):
    payload = completed_payload()
    payload["results"][0]["scores"][field] = value
    with pytest.raises(ValidationError):
        parse(payload)


def test_rejects_units_that_do_not_sum_to_result():
    payload = completed_payload()
    payload["results"][0]["units"][0]["scores"]["invalid_outputs"] = 0
    with pytest.raises(ValidationError, match="unit"):
        parse(payload)


def test_pipeline_cannot_silently_omit_one_declared_source():
    payload = experiment_payload()
    other = copy.deepcopy(payload["configurations"][0])
    other.update(key="other-model", pipeline="other")
    payload["configurations"].append(other)
    outcome = copy.deepcopy(payload["results"][0])
    outcome["key"] = "other-model"
    payload["results"].append(outcome)
    with pytest.raises(ValidationError, match="inventory"):
        parse(payload)


def test_pipeline_cannot_mix_model_revisions_between_languages():
    payload = experiment_payload()
    payload["configurations"][1]["models"][0] = {
        "identifier": "different-checkpoint",
        "revision": "f" * 40,
        "sha256": "0" * 64,
    }
    with pytest.raises(ValidationError, match="pipeline"):
        parse(payload)


def test_pairs_must_use_identical_document_ids_and_gold_counts():
    payload = completed_payload()
    payload["configurations"] = [payload["configurations"][0]]
    payload["results"] = [payload["results"][0]]
    duplicate = copy.deepcopy(payload["configurations"][0])
    duplicate.update(key="other-model", pipeline="other")
    payload["configurations"].append(duplicate)
    outcome = copy.deepcopy(payload["results"][0])
    outcome.update(
        key="other-model",
        status="planned",
        scores=None,
        measurements=None,
        units=[],
        evaluated_examples=0,
        failed_examples=0,
    )
    payload["results"].append(outcome)
    run = parse(payload)
    outcome.update(copy.deepcopy(payload["results"][0]))
    outcome.update(
        key="other-model",
        provenance_sha256=run.provenance_digest(run.configurations[1]),
    )
    outcome["units"][0]["example_id"] = "different-document"
    with pytest.raises(ValidationError, match="paired"):
        parse(payload)


@pytest.mark.parametrize("status", ["failed", "unsupported"])
def test_explicit_noncompletion_remains_visible(status):
    payload = experiment_payload()
    payload["results"][0].update(
        status=status, reason="Fixture cannot run this language"
    )
    assert parse(payload).results[0].status == status


def test_end_to_end_requires_pinned_gazetteer_and_separate_count_type():
    payload = completed_payload()
    payload["protocol"]["task"] = "end_to_end"
    with pytest.raises(ValidationError, match="gazetteer"):
        parse(payload)
    for config in payload["configurations"]:
        config["gazetteer"] = copy.deepcopy(config["models"][0])
    payload["results"][0]["status"] = "planned"
    measured = payload["results"][0]
    counts, units = measured.pop("scores"), measured.pop("units")
    measurements = measured.pop("measurements")
    measured.update(
        scores=None, measurements=None, evaluated_examples=0, failed_examples=0
    )
    run = parse(payload)
    counts["task"] = "end_to_end"
    for unit in units:
        unit["scores"]["task"] = "end_to_end"
    measured.update(
        status="complete",
        scores=counts,
        units=units,
        measurements=measurements,
        evaluated_examples=1,
        failed_examples=1,
        provenance_sha256=run.provenance_digest(run.configurations[0]),
    )
    assert parse(payload).results[0].scores.task == "end_to_end"


def test_no_complete_scores_are_available_for_planned_result():
    result = parse(experiment_payload()).results[0]
    with pytest.raises(ValueError, match="no complete scores"):
        result.completed_scores()


def test_model_specific_parameters_and_reviewed_custom_code_are_pinned():
    payload = experiment_payload()
    original = parse(payload)
    config = payload["configurations"][0]
    config["parameters"] = {
        "max_length": 512,
        "decode": {"temperature": 0.0},
        "template": "location-only-v1",
    }
    config["custom_code"] = [
        {
            "code": copy.deepcopy(config["models"][0]),
            "review": {"identifier": "review.txt", "revision": "1", "sha256": "8" * 64},
        }
    ]
    updated = parse(payload)
    assert original.provenance_digest(
        original.configurations[0]
    ) != updated.provenance_digest(updated.configurations[0])
    del config["custom_code"][0]["review"]
    with pytest.raises(ValidationError, match="review"):
        parse(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nested_parameters_reject_nonfinite_numbers(value):
    payload = experiment_payload()
    payload["configurations"][0]["parameters"] = {"nested": {"temperature": value}}
    with pytest.raises(ValidationError, match="parameters"):
        parse(payload)


def test_resolution_pairs_cannot_change_eligible_denominators():
    payload = completed_payload()
    payload["configurations"] = payload["configurations"][:1]
    payload["results"] = payload["results"][:1]
    payload["protocol"]["task"] = "gold_span_resolution"
    payload["configurations"][0]["gazetteer"] = payload["protocol"]["dataset"]
    count = {
        "task": "gold_span_resolution",
        "gold_spans": 3,
        "resolved": 2,
        "abstained": 1,
        "invalid_outputs": 0,
        "exact_id_eligible": 3,
        "exact_id_correct": 1,
        "coordinate_eligible": 3,
        "within_1km": 1,
        "within_10km": 1,
        "within_50km": 1,
        "candidate_eligible": 3,
        "candidate_found": 2,
    }
    units = [
        {
            "example_id": "first",
            "failed": False,
            "scores": dict(
                count,
                gold_spans=2,
                abstained=0,
                exact_id_eligible=2,
                coordinate_eligible=2,
                candidate_eligible=2,
            ),
        },
        {
            "example_id": "second",
            "failed": True,
            "scores": dict(
                count,
                gold_spans=1,
                resolved=0,
                exact_id_eligible=1,
                exact_id_correct=0,
                coordinate_eligible=1,
                within_1km=0,
                within_10km=0,
                within_50km=0,
                candidate_eligible=1,
                candidate_found=0,
            ),
        },
    ]
    result = payload["results"][0]
    result.update(
        status="planned",
        scores=None,
        units=[],
        measurements=None,
        evaluated_examples=0,
        failed_examples=0,
    )
    duplicate = copy.deepcopy(payload["configurations"][0])
    duplicate.update(key="other", pipeline="other")
    payload["configurations"].append(duplicate)
    other = copy.deepcopy(result)
    other["key"] = "other"
    payload["results"].append(other)
    run = parse(payload)
    for index, outcome in enumerate(payload["results"]):
        outcome.update(
            status="complete",
            scores=copy.deepcopy(count),
            units=copy.deepcopy(units),
            evaluated_examples=1,
            failed_examples=1,
            measurements=completed_payload()["results"][0]["measurements"],
            provenance_sha256=run.provenance_digest(run.configurations[index]),
        )
    other["scores"]["coordinate_eligible"] = 2
    other["units"][0]["scores"]["coordinate_eligible"] = 1
    with pytest.raises(ValidationError, match="paired"):
        parse(payload)


def test_failed_units_cannot_contribute_successful_predictions():
    payload = completed_payload()
    payload["results"][0]["units"][0]["failed"] = True
    payload["results"][0].update(evaluated_examples=0, failed_examples=2)
    with pytest.raises(ValidationError, match="failed"):
        parse(payload)


@pytest.mark.parametrize("task", ["recognition", "end_to_end"])
def test_failed_span_units_retain_invalid_output_penalties(task):
    from scripts.benchmark_protocol.schema import Unit

    unit = Unit.model_validate(
        {
            "example_id": "invalid",
            "failed": True,
            "scores": {
                "task": task,
                "gold_spans": 1,
                "predicted_spans": 1,
                "true_positive": 0,
                "false_positive": 1,
                "false_negative": 1,
                "invalid_outputs": 1,
            },
        }
    )
    assert unit.scores.invalid_outputs == 1


def test_failed_resolution_cannot_return_even_an_incorrect_valid_prediction():
    from scripts.benchmark_protocol.schema import Unit

    scores = {
        "task": "gold_span_resolution",
        "gold_spans": 1,
        "resolved": 1,
        "abstained": 0,
        "invalid_outputs": 0,
        "exact_id_eligible": 1,
        "exact_id_correct": 0,
        "coordinate_eligible": 1,
        "within_1km": 0,
        "within_10km": 0,
        "within_50km": 0,
        "candidate_eligible": 1,
        "candidate_found": 1,
    }
    with pytest.raises(ValidationError, match="failed"):
        Unit.model_validate(
            {"example_id": "ranking-failed", "failed": True, "scores": scores}
        )
