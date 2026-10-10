"""Freeze-plan validation on a synthetic plan, judged by the shared protocol validator."""

import copy

import pytest
from pydantic import ValidationError

from scripts.benchmark_protocol.schema import Experiment
from scripts.embedding_resolution.models import MODELS
from scripts.embedding_resolution.protocol import FreezePlan, build_experiment

MINILM_REVISION = "d678769f195c194ffdc6f3735a9360118a855b43"
JINA_REVISION = "dd76d535f5447ca3897a9c893fb1e612ead98192"
# Five registered models under two embedding policies, plus the baseline, name
# 11 pairs. MiniLM's historical 0.6 similarity record adds a twelfth pipeline.
PIPELINES = 12
SOURCES = 2


def artifact(identifier, revision, digit):
    """A synthetic artifact; digests are fixed fixture values, not real files."""
    return {"identifier": identifier, "revision": revision, "sha256": digit * 64}


def calibrated(model, policy):
    """A development-calibrated record for a registered model under one policy."""
    return {
        "model": model,
        "policy": policy,
        "min_similarity": 0.5,
        "origin": "development_calibrated",
        "calibration_sha256": "5" * 64,
        "note": f"Development calibration of {model} under {policy}.",
    }


def plan_payload():
    """A complete plan: every registered model under both policies, and the baseline."""
    protocol = {
        "schema_version": "1.0",
        "task": "gold_span_resolution",
        "code_revision": "c" * 40,
        "dependency_lock_sha256": "d" * 64,
        "dataset": artifact("public-benchmark-fixture", "a" * 40, "b"),
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
    return {
        "protocol": protocol,
        "gazetteer": artifact("geonames", "e" * 40, "f"),
        "context": {
            "token_limit": 256,
            "tokenizer": artifact("tokenizer-fixture", "7" * 40, "8"),
        },
        "retrieval": {
            "max_tiers": 3,
            "search_methods": ["exact", "phrase", "partial", "fuzzy"],
            "candidate_limit": 10000,
        },
        "models": {
            model.key: artifact(model.repository, model.revision, str(index + 3))
            for index, model in enumerate(MODELS)
        },
        "reviewed_code": {
            "jina-v5-text-small": [
                {
                    "code": artifact(
                        "jinaai/jina-embeddings-v5-text-small", JINA_REVISION, "9"
                    ),
                    "review": artifact("review-fixture", "a" * 40, "c"),
                }
            ]
        },
        "sources": [
            {
                "language": "en",
                "source_config": "geovirus",
                "examples": 2,
                "gold_spans": 3,
                "sample_sha256": "1" * 64,
            },
            {
                "language": "fr",
                "source_config": "hipe2020-fr",
                "examples": 8,
                "gold_spans": 9,
                "sample_sha256": "2" * 64,
            },
        ],
        "thresholds": [
            {
                "model": "geo-minilm",
                "policy": "similarity",
                "min_similarity": 0.31,
                "origin": "development_calibrated",
                "calibration_sha256": "5" * 64,
                "note": "Development calibration of MiniLM without a prior.",
            },
            {
                "model": "geo-minilm",
                "policy": "similarity",
                "min_similarity": 0.6,
                "origin": "historical",
                "calibration_sha256": None,
                "note": "Resolver constructor default, kept for comparison.",
            },
            {
                "model": None,
                "policy": "population_only",
                "min_similarity": 0.0,
                "origin": "structural",
                "calibration_sha256": None,
                "note": "Most populous candidate, no similarity cutoff.",
            },
            *[
                calibrated(model.key, policy)
                for model in MODELS
                for policy in ("similarity", "population")
                if (model.key, policy) != ("geo-minilm", "similarity")
            ],
        ],
        "batch_size": 32,
        "seed": 0,
        "provenance_note": "Synthetic plan for offline validation only.",
    }


def valid_plan():
    return FreezePlan.model_validate(plan_payload())


def _first(experiment, parameter, value):
    """The first configuration whose parameter has the given value."""
    return next(
        config
        for config in experiment.configurations
        if config.parameters[parameter] == value
    )


def test_a_plan_expands_into_one_planned_configuration_per_pipeline_and_source():
    experiment = build_experiment(valid_plan())

    assert isinstance(experiment, Experiment)
    assert len(experiment.configurations) == PIPELINES * SOURCES
    assert {result.status for result in experiment.results} == {"planned"}
    assert len({config.pipeline for config in experiment.configurations}) == PIPELINES


def test_every_pipeline_covers_every_source_slice():
    experiment = build_experiment(valid_plan())

    cells = {
        (config.pipeline, config.language, config.source_config)
        for config in experiment.configurations
    }
    assert len(cells) == PIPELINES * SOURCES


def test_a_historical_zero_six_is_labelled_and_its_pipeline_name_says_so():
    experiment = build_experiment(valid_plan())

    historical = [
        config
        for config in experiment.configurations
        if config.parameters["threshold_origin"] == "historical"
    ]
    assert {config.thresholds["min_similarity"] for config in historical} == {0.6}
    assert all("historical" in config.pipeline for config in historical)


def test_the_population_only_baseline_uses_the_gazetteer_and_no_model_weights():
    experiment = build_experiment(valid_plan())

    baseline = _first(experiment, "policy", "population_only")
    assert (
        baseline.models[0].identifier,
        baseline.parameters["model"],
        baseline.parameters["population_weight"],
        baseline.custom_code,
    ) == ("geonames", None, None, [])


def test_policies_record_the_prior_weight_they_apply():
    experiment = build_experiment(valid_plan())

    weights = {
        config.parameters["policy"]: config.parameters["population_weight"]
        for config in experiment.configurations
        if config.parameters["policy"] != "population_only"
    }
    assert weights == {"similarity": 0.0, "population": 0.3}


def test_a_calibrated_record_relabelled_as_population_is_refused():
    payload = plan_payload()
    payload["thresholds"][0]["policy"] = "population"

    with pytest.raises(ValidationError, match="exactly one development-calibrated"):
        FreezePlan.model_validate(payload)


def test_a_missing_calibrated_cell_is_refused_even_with_a_historical_row_for_it():
    payload = plan_payload()
    payload["thresholds"] = [
        record
        for record in payload["thresholds"]
        if (record["model"], record["policy"], record["origin"])
        != ("geo-minilm", "similarity", "development_calibrated")
    ]
    assert any(
        (record["model"], record["policy"]) == ("geo-minilm", "similarity")
        and record["origin"] == "historical"
        for record in payload["thresholds"]
    )

    with pytest.raises(ValidationError, match="every registered model"):
        FreezePlan.model_validate(payload)


def test_a_historical_row_relabelled_as_population_cannot_fill_that_cell():
    payload = plan_payload()
    payload["thresholds"] = [
        record
        for record in payload["thresholds"]
        if (record["model"], record["policy"], record["origin"])
        != ("geo-minilm", "population", "development_calibrated")
    ]
    historical = next(
        record for record in payload["thresholds"] if record["origin"] == "historical"
    )
    # 0.0 is the registered historical population setting, so the record itself is
    # accepted and only the missing calibrated cell can refuse the plan.
    historical["policy"] = "population"
    historical["min_similarity"] = 0.0

    with pytest.raises(ValidationError, match="every registered model"):
        FreezePlan.model_validate(payload)


def test_duplicate_calibrated_rows_for_another_pair_are_refused():
    payload = plan_payload()
    duplicate = calibrated("qwen3-embedding-0.6b", "population")
    duplicate["min_similarity"] = 0.4
    payload["thresholds"].append(duplicate)

    with pytest.raises(ValidationError, match="exactly one development-calibrated"):
        FreezePlan.model_validate(payload)


def test_a_historical_extra_is_frozen_beside_the_calibrated_cell_it_shares():
    payload = plan_payload()
    payload["thresholds"].append(
        {
            "model": "geo-minilm",
            "policy": "population",
            "min_similarity": 0.0,
            "origin": "historical",
            "calibration_sha256": None,
            "note": "Benchmark prior run, kept for comparison.",
        }
    )
    experiment = build_experiment(FreezePlan.model_validate(payload))

    extras = [
        config
        for config in experiment.configurations
        if config.parameters["threshold_origin"] == "historical"
        and config.parameters["policy"] == "population"
    ]
    assert (
        len(experiment.configurations),
        {config.thresholds["min_similarity"] for config in extras},
        len({config.pipeline for config in experiment.configurations}),
    ) == ((PIPELINES + 1) * SOURCES, {0.0}, PIPELINES + 1)


def test_model_configurations_record_the_pinned_recipe_and_context_budget():
    experiment = build_experiment(valid_plan())

    minilm = _first(experiment, "model", "geo-minilm")
    assert (
        minilm.models[0].revision,
        minilm.parameters["pooling"],
        minilm.parameters["context_token_limit"],
        minilm.parameters["context_tokenizer"]["identifier"],
        bool(minilm.parameters["attribute_map"]["name"]),
    ) == (MINILM_REVISION, "mean", 256, "tokenizer-fixture", True)


def test_each_planned_row_carries_no_execution_evidence():
    experiment = build_experiment(valid_plan())

    evidence = {
        (
            result.provenance_sha256,
            result.measurements,
            result.scores,
            result.raw_predictions,
            result.evaluated_examples + result.failed_examples,
        )
        for result in experiment.results
    }
    assert evidence == {(None, None, None, None, 0)}


def test_a_calibrated_threshold_must_name_its_development_digest():
    payload = plan_payload()
    payload["thresholds"][0]["calibration_sha256"] = None

    with pytest.raises(ValidationError, match="calibration digest"):
        FreezePlan.model_validate(payload)


def test_a_historical_value_is_only_accepted_for_the_model_that_used_it():
    payload = plan_payload()
    payload["thresholds"][1]["model"] = "jina-v5-text-small"
    payload["thresholds"][1]["origin"] = "historical"
    payload["models"]["jina-v5-text-small"] = artifact(
        "jinaai/jina-embeddings-v5-text-small", JINA_REVISION, "4"
    )

    with pytest.raises(ValidationError, match="not registered for this model"):
        FreezePlan.model_validate(payload)


def test_a_historical_value_is_refused_under_a_policy_that_never_used_it():
    payload = plan_payload()
    payload["thresholds"][1]["policy"] = "population"

    with pytest.raises(
        ValidationError, match="not registered for this model and policy"
    ):
        FreezePlan.model_validate(payload)


@pytest.mark.parametrize("value", [1.5, 0.333, -1.2])
def test_a_calibrated_threshold_off_the_calibration_grid_is_refused(value):
    payload = plan_payload()
    payload["thresholds"][0]["min_similarity"] = value

    with pytest.raises(ValidationError, match="calibration grid"):
        FreezePlan.model_validate(payload)


def test_an_unregistered_model_key_is_refused_by_the_plan():
    payload = plan_payload()
    payload["thresholds"][0]["model"] = "unregistered"

    with pytest.raises(ValidationError, match="outside the registry"):
        FreezePlan.model_validate(payload)


def test_a_baseline_with_a_model_or_a_cutoff_is_refused():
    payload = plan_payload()
    payload["thresholds"][2]["model"] = "geo-minilm"

    with pytest.raises(ValidationError, match="uses no embedding model"):
        FreezePlan.model_validate(payload)


def test_a_recognition_task_is_refused():
    payload = plan_payload()
    payload["protocol"]["task"] = "recognition"

    with pytest.raises(ValidationError, match="gold_span_resolution task"):
        FreezePlan.model_validate(payload)


@pytest.mark.parametrize(
    "selection",
    [
        {"stage": "final", "split": "test", "selection_sha256": "e" * 64},
        {
            "stage": "screening",
            "split": "development",
            "threshold_selection": "fixed_before_evaluation",
        },
        {
            "stage": "final",
            "split": "test",
            "threshold_selection": "fixed_before_evaluation",
            "selection_sha256": "e" * 64,
        },
    ],
)
def test_a_plan_that_is_not_development_screening_is_refused(selection):
    payload = plan_payload()
    payload["protocol"].update(selection)

    with pytest.raises(ValidationError, match="development screening only"):
        FreezePlan.model_validate(payload)


def test_a_weight_artifact_that_differs_from_the_registry_pin_is_refused():
    payload = plan_payload()
    payload["models"]["geo-minilm"]["revision"] = "0" * 40

    with pytest.raises(ValueError, match="differs from the registry pin"):
        build_experiment(FreezePlan.model_validate(payload))


def test_a_model_without_a_weight_artifact_is_refused_by_the_plan():
    payload = plan_payload()
    del payload["models"]["geo-minilm"]

    with pytest.raises(ValidationError, match="weight artifact"):
        FreezePlan.model_validate(payload)


def test_custom_code_models_are_refused_until_their_code_is_reviewed():
    payload = plan_payload()
    del payload["reviewed_code"]

    with pytest.raises(ValueError, match="reviewed and pinned before the freeze"):
        build_experiment(FreezePlan.model_validate(payload))


def test_reviewed_custom_code_is_carried_into_each_of_that_models_configurations():
    experiment = build_experiment(valid_plan())

    jina = [
        config
        for config in experiment.configurations
        if config.parameters["model"] == "jina-v5-text-small"
    ]
    assert (
        len(jina),
        [len(config.custom_code) for config in jina],
        jina[0].parameters["task"],
    ) == (4, [1, 1, 1, 1], "retrieval")


@pytest.mark.parametrize(
    ("identifier", "revision"),
    [
        ("jinaai/jina-embeddings-v5-text-small", "0" * 40),
        ("someone-else/jina-embeddings-v5-text-small", JINA_REVISION),
    ],
)
def test_a_review_of_code_that_is_not_the_registry_pin_is_refused(identifier, revision):
    payload = plan_payload()
    payload["reviewed_code"] = {
        "jina-v5-text-small": [
            {
                "code": artifact(identifier, revision, "9"),
                "review": artifact("review-fixture", "a" * 40, "c"),
            }
        ]
    }

    with pytest.raises(
        ValueError, match=r"reviewed code .* differs from the registry pin"
    ):
        build_experiment(FreezePlan.model_validate(payload))


def test_reviewed_code_for_an_unknown_model_is_refused():
    payload = plan_payload()
    payload["reviewed_code"] = {
        "unregistered": [
            {
                "code": artifact("x", "a" * 40, "b"),
                "review": artifact("y", "c" * 40, "d"),
            }
        ]
    }

    with pytest.raises(ValidationError, match="outside the registry"):
        FreezePlan.model_validate(payload)


def test_a_duplicate_threshold_record_is_refused():
    payload = plan_payload()
    payload["thresholds"].append(copy.deepcopy(payload["thresholds"][0]))

    with pytest.raises(ValidationError, match="duplicate threshold record"):
        FreezePlan.model_validate(payload)


def test_a_duplicate_source_is_refused():
    payload = plan_payload()
    payload["sources"].append(copy.deepcopy(payload["sources"][0]))

    with pytest.raises(ValidationError, match="duplicate source"):
        FreezePlan.model_validate(payload)


def test_a_gazetteer_without_an_attribute_map_is_refused():
    payload = plan_payload()
    payload["gazetteer"]["identifier"] = "unmapped-gazetteer"

    with pytest.raises(ValidationError, match="attribute map"):
        FreezePlan.model_validate(payload)


def test_the_expanded_inventory_round_trips_through_the_shared_validator():
    experiment = build_experiment(valid_plan())

    reparsed = Experiment.model_validate_json(experiment.model_dump_json())

    assert reparsed == experiment


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("max_tiers", 2),
        ("candidate_limit", 5000),
        ("search_methods", ["phrase", "exact", "partial", "fuzzy"]),
    ],
)
def test_plans_differing_only_in_a_retrieval_setting_get_different_digests(
    setting, value
):
    base = build_experiment(valid_plan())
    payload = plan_payload()
    payload["retrieval"][setting] = value
    changed = build_experiment(FreezePlan.model_validate(payload))

    assert base.provenance_digest(base.configurations[0]) != changed.provenance_digest(
        changed.configurations[0]
    )


def test_a_search_method_listed_twice_is_refused():
    payload = plan_payload()
    payload["retrieval"]["search_methods"] = ["exact", "exact"]

    with pytest.raises(ValidationError, match="listed once"):
        FreezePlan.model_validate(payload)


def test_an_embedding_threshold_without_a_model_key_is_refused():
    payload = plan_payload()
    payload["thresholds"][0]["model"] = None

    with pytest.raises(ValidationError, match="needs a model key"):
        FreezePlan.model_validate(payload)


def test_a_plan_with_only_the_baseline_is_refused():
    payload = plan_payload()
    payload["thresholds"] = [
        record
        for record in payload["thresholds"]
        if record["policy"] == "population_only"
    ]

    with pytest.raises(ValidationError, match="every registered model"):
        FreezePlan.model_validate(payload)


@pytest.mark.parametrize(
    "omitted",
    [
        pytest.param(lambda record: record["model"] == "bge-m3", id="registered-model"),
        pytest.param(
            lambda record: (
                record["model"] == "qwen3-embedding-4b"
                and record["policy"] == "population"
            ),
            id="embedding-policy",
        ),
        pytest.param(
            lambda record: record["policy"] == "population_only", id="baseline"
        ),
    ],
)
def test_a_plan_omitting_a_model_a_policy_or_the_baseline_is_refused(omitted):
    payload = plan_payload()
    payload["thresholds"] = [
        record for record in payload["thresholds"] if not omitted(record)
    ]

    with pytest.raises(ValidationError, match="every registered model"):
        FreezePlan.model_validate(payload)


def test_an_extra_weight_artifact_outside_the_registry_is_refused():
    payload = plan_payload()
    payload["models"]["unregistered"] = artifact("x", "a" * 40, "b")

    with pytest.raises(ValidationError, match="weight artifact for each registered"):
        FreezePlan.model_validate(payload)
