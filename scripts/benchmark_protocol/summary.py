"""Count-weighted micro, equal-language macro, and document bootstrap summaries."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from statistics import fmean

from scripts.benchmark_protocol.schema import (
    Configuration,
    Experiment,
    Result,
    Scores,
    SpanCounts,
    require,
)
from scripts.panx_benchmark.metrics import Counts, ratio


def metrics(counts: Scores) -> dict[str, float]:
    """Calculate metrics using explicit eligible denominators, never resolved-only."""
    if isinstance(counts, SpanCounts):
        values = Counts(
            true_positive=counts.true_positive,
            false_positive=counts.false_positive,
            false_negative=counts.false_negative,
        ).scores()
        return {key: float(values[key]) for key in ("precision", "recall", "f1")}
    return {
        "exact_id_accuracy": ratio(counts.exact_id_correct, counts.exact_id_eligible),
        "accuracy_at_1km": ratio(counts.within_1km, counts.coordinate_eligible),
        "accuracy_at_10km": ratio(counts.within_10km, counts.coordinate_eligible),
        "accuracy_at_50km": ratio(counts.within_50km, counts.coordinate_eligible),
        "coverage": ratio(counts.resolved, counts.gold_spans),
        "abstention_rate": ratio(counts.abstained, counts.gold_spans),
        "invalid_output_rate": ratio(counts.invalid_outputs, counts.gold_spans),
        "candidate_recall": ratio(counts.candidate_found, counts.candidate_eligible),
    }


def _sum_counts(rows: list[Scores]) -> Scores:
    """Pool compatible count records before calculating ratios."""
    require(bool(rows), "Cannot pool an empty count inventory")
    require(
        all(row.task == rows[0].task for row in rows), "Cannot pool different tasks"
    )
    payload = rows[0].model_dump()
    for key in payload.keys() - {"task"}:
        payload[key] = sum(getattr(row, key) for row in rows)
    return type(rows[0]).model_validate(payload)


def _interval(values: list[float]) -> list[float]:
    """Return the empirical inverse-CDF 2.5% and 97.5% percentiles."""
    ordered = sorted(values)
    size = len(ordered)
    return [
        ordered[max(0, (25 * size + 999) // 1000 - 1)],
        ordered[(975 * size + 999) // 1000 - 1],
    ]


def bootstrap(
    units: list[Scores], *, seed: int, resamples: int
) -> dict[str, list[float]]:
    """Resample whole examples with replacement; zero denominators score zero."""
    require(resamples > 0, "bootstrap resamples must be positive")
    require(bool(units), "bootstrap needs document units")
    _sum_counts(units)
    rng = random.Random(seed)  # noqa: S311 - seeded statistical resampling
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        for metric, value in metrics(
            _sum_counts(rng.choices(units, k=len(units)))
        ).items():
            samples[metric].append(value)
    return {metric: _interval(values) for metric, values in samples.items()}


def inventory_summary(experiment: Experiment) -> dict:
    """Describe all configurations without running models or bootstrap computation."""
    from scripts.panx_benchmark.data import target_languages

    languages = sorted({config.language for config in experiment.configurations})
    rows = {result.key: result for result in experiment.results}
    return {
        "schema_version": experiment.protocol.schema_version,
        "task": experiment.protocol.task,
        "status_counts": dict(Counter(result.status for result in experiment.results)),
        "configuration_count": len(experiment.configurations),
        "inventory": _inventory_rows(experiment.configurations, rows),
        "included_languages": languages,
        "missing_target_languages": sorted(set(target_languages()) - set(languages)),
        "configuration_digests": {
            config.key: experiment.provenance_digest(config)
            for config in experiment.configurations
        },
    }


def aggregate(experiment: Experiment) -> dict:
    """Keep pipelines/seeds separate and make every omitted score visible."""
    experiment = Experiment.model_validate(experiment.model_dump())
    report = inventory_summary(experiment)
    rows = {result.key: result for result in experiment.results}
    groups: dict[tuple[str, int], list[Configuration]] = defaultdict(list)
    for config in experiment.configurations:
        groups[(config.pipeline, config.seed)].append(config)
    pipelines: dict = defaultdict(dict)
    for (pipeline, seed), configs in sorted(groups.items()):
        pipelines[pipeline][str(seed)] = _pipeline_summary(experiment, configs, rows)
    report["pipelines"] = dict(pipelines)
    return report


def _pipeline_summary(
    experiment: Experiment, configs: list[Configuration], rows: dict[str, Result]
) -> dict:
    """Calculate only completed cells without hiding incomplete membership."""
    completed = [config for config in configs if rows[config.key].status == "complete"]
    inventory = _inventory_rows(configs, rows)
    if not completed:
        return {"complete": False, "languages": [], "inventory": inventory}
    units = {
        (config.language, config.source_config): _ordered_units(rows[config.key])
        for config in completed
    }
    summary = _group_summary(experiment, units)
    summary["complete"] = len(completed) == len(configs)
    summary["inventory"] = inventory
    return summary


def _ordered_units(result: Result) -> list[Scores]:
    """Sort by stable example ID before generating paired resamples."""
    ordered = sorted(result.units, key=lambda unit: unit.example_id)
    return [unit.scores for unit in ordered]


def _language_counts(units: dict[tuple[str, str], list[Scores]]) -> dict[str, Scores]:
    """Pool all included source configurations within each language."""
    grouped: dict[str, list[Scores]] = defaultdict(list)
    for (language, _source), rows in sorted(units.items()):
        grouped[language].extend(rows)
    return {language: _sum_counts(rows) for language, rows in grouped.items()}


def _macro(rows: dict[str, Scores]) -> dict[str, float]:
    """Give each included language equal weight after pooling its source counts."""
    scores = [metrics(counts) for counts in rows.values()]
    return {key: fmean(row[key] for row in scores) for key in scores[0]}


def _group_summary(
    experiment: Experiment, units: dict[tuple[str, str], list[Scores]]
) -> dict:
    """Report point estimates and stratified paired-document confidence intervals."""
    counts = _language_counts(units)
    return {
        "languages": sorted(counts),
        "per_language": {
            language: {"counts": row.model_dump(), "metrics": metrics(row)}
            for language, row in counts.items()
        },
        "macro": _macro(counts),
        "micro": metrics(_sum_counts(list(counts.values()))),
        "uncertainty": _group_intervals(experiment, units),
    }


def _group_intervals(
    experiment: Experiment, units: dict[tuple[str, str], list[Scores]]
) -> dict:
    """Use a stable RNG per source so matching pipeline comparisons stay paired."""
    protocol = experiment.protocol
    generators = {key: _generator(protocol.bootstrap_seed, key) for key in units}
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for _ in range(protocol.bootstrap_resamples):
        sampled = {
            key: generators[key].choices(rows, k=len(rows))
            for key, rows in units.items()
        }
        _append_sample(samples, _sample_metrics(sampled))
    return {
        "method": protocol.uncertainty,
        "seed": protocol.bootstrap_seed,
        "resamples": protocol.bootstrap_resamples,
        "intervals": _sample_intervals(samples),
    }


def _generator(seed: int, key: tuple[str, str]) -> random.Random:
    """Create a reproducible non-security random generator for one source."""
    digest = hashlib.sha256(json_seed(seed, key).encode()).digest()
    return random.Random(digest)  # noqa: S311 - statistical bootstrap, not cryptography


def _sample_metrics(
    units: dict[tuple[str, str], list[Scores]],
) -> dict[str, dict[str, float]]:
    """Recompute every metric on a single stratified bootstrap draw."""
    counts = _language_counts(units)
    return {
        **{f"language:{language}": metrics(row) for language, row in counts.items()},
        "macro": _macro(counts),
        "micro": metrics(_sum_counts(list(counts.values()))),
    }


def _append_sample(
    samples: dict[str, dict[str, list[float]]], values: dict[str, dict[str, float]]
) -> None:
    """Retain each resampled statistic for percentile calculation."""
    for group, scores in values.items():
        for metric, value in scores.items():
            samples[group][metric].append(value)


def _sample_intervals(samples: dict[str, dict[str, list[float]]]) -> dict:
    """Convert resampled statistics to confidence intervals."""
    return {
        group: {metric: _interval(values) for metric, values in scores.items()}
        for group, scores in samples.items()
    }


def json_seed(seed: int, key: tuple[str, str]) -> str:
    """Make unambiguous seed material for each language/source stratum."""
    import json

    return json.dumps([seed, *key], separators=(",", ":"))


def _inventory_row(config: Configuration, result: Result) -> dict:
    """Keep each source's status, failure reason, and score membership visible."""
    return {
        "key": config.key,
        "pipeline": config.pipeline,
        "seed": config.seed,
        "language": config.language,
        "source_config": config.source_config,
        "sample_sha256": config.sample_sha256,
        "examples": config.examples,
        "gold_spans": config.gold_spans,
        "status": result.status,
        "reason": result.reason,
        "evaluated_examples": result.evaluated_examples,
        "failed_examples": result.failed_examples,
        "contributes_to_metrics": result.status == "complete",
    }


def _inventory_rows(
    configs: list[Configuration], rows: dict[str, Result]
) -> list[dict]:
    """Return all declared cells, including unsupported and failed configurations."""
    return [
        _inventory_row(config, rows[config.key])
        for config in sorted(configs, key=lambda item: item.key)
    ]
