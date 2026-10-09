"""A synthetic zh file, scored by the adapter, must satisfy the shared protocol."""

import hashlib

import pytest
from pydantic import ValidationError

from scripts.benchmark_protocol.schema import Experiment
from scripts.multiconer_benchmark.conll import parse_conll, require_no_invalid_records
from scripts.multiconer_benchmark.label_policy import label_mapping
from scripts.multiconer_benchmark.scoring import score_sentence, sum_counts

FIXTURE = (
    "# id zh-1\tdomain=zh\n"
    "我 _ _ O\n在 _ _ O\n北京 _ _ B-HumanSettlement\n的 _ _ O\n"
    "故宫 _ _ B-Facility\n参观 _ _ O\n\n"
    "# id zh-2\tdomain=zh\n"
    "雨 _ _ O\n下 _ _ O\n\n"
)


def _protocol():
    artifact = {
        "identifier": "synthetic-multiconer-fixture",
        "revision": "a" * 40,
        "sha256": "b" * 64,
    }
    return artifact, {
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
        "uncertainty": "marginal_stratified_document_bootstrap_95_percent",
        "bootstrap_seed": 42,
        "bootstrap_resamples": 1000,
    }


def _payload():
    artifact, protocol = _protocol()
    parsed = parse_conll(FIXTURE, expected_domain="zh")
    gold_total = sum(len(s.location_spans()) for s in parsed.sentences)
    configuration = {
        "key": "zh",
        "pipeline": "fixture",
        "language": "zh",
        "source_config": "zh",
        "examples": parsed.record_count,
        "gold_spans": gold_total,
        "sample_sha256": hashlib.sha256(FIXTURE.encode("utf-8")).hexdigest(),
        "models": [artifact],
        "label_mapping": label_mapping(),
        "thresholds": {"recognition": 0.5},
        "batch_size": 1,
        "seed": 0,
        "language_support": "documented",
        "training_overlap": "unknown",
        "provenance_note": "Synthetic fixture, no training claim.",
        "gazetteer": None,
    }
    return {
        "protocol": protocol,
        "configurations": [configuration],
        "results": [
            {
                "key": "zh",
                "status": "planned",
                "reason": None,
                "provenance_sha256": None,
                "measurements": None,
                "scores": None,
                "raw_predictions": None,
                "evaluated_examples": 0,
                "failed_examples": 0,
            }
        ],
    }, parsed


def _complete():
    payload, parsed = _payload()
    run = Experiment.model_validate(payload)
    units, rows = [], []
    predictions = {"zh-1": [(4, 6), (9, 11)], "zh-2": []}
    for sentence in parsed.sentences:
        scores = score_sentence(
            sentence.text, sentence.location_spans(), predictions[sentence.sample_id]
        )
        rows.append(scores)
        units.append(
            {
                "example_id": sentence.sample_id,
                "failed": False,
                "scores": scores.model_dump(mode="json"),
            }
        )
    total = sum_counts(rows)
    payload["results"][0].update(
        status="complete",
        provenance_sha256=run.provenance_digest(run.configurations[0]),
        scores=total.model_dump(mode="json"),
        units=units,
        evaluated_examples=parsed.record_count,
        failed_examples=0,
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


def test_adapter_output_is_a_valid_protocol_inventory_and_completed_result():
    experiment = Experiment.model_validate(_complete())
    result = experiment.results[0]
    assert result.scores.gold_spans == 2
    assert result.scores.true_positive == 2
    assert result.scores.false_positive == 0


def test_a_unit_count_that_disagrees_with_the_total_is_rejected():
    payload = _complete()
    payload["results"][0]["units"][0]["scores"]["true_positive"] = 1
    with pytest.raises(ValidationError):
        Experiment.model_validate(payload)


def test_sources_with_invalid_records_cannot_be_frozen_as_complete_inventory():
    polluted = FIXTURE + "Rome _ _ B-HumanSettlement\n\n"
    report = parse_conll(polluted, expected_domain="zh")
    assert report.invalid
    with pytest.raises(ValueError, match="invalid records"):
        require_no_invalid_records(report)
    require_no_invalid_records(parse_conll(FIXTURE, expected_domain="zh"))
