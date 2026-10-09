"""Strict JSON contract for comparable, explicitly accounted benchmark runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from scripts.panx_benchmark.data import target_languages

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Commit = Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
Revision = Annotated[
    str,
    Field(
        pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64}|v?[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:[-+][0-9A-Za-z.-]+)?)$"
    ),
]
Count = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(gt=0)]
Measurement = Annotated[float, Field(ge=0)]
Task = Literal["recognition", "gold_span_resolution", "end_to_end"]


def require(condition: bool, message: str) -> None:  # noqa: FBT001 - internal assertion predicate
    """Reject a contract violation with a useful validation message."""
    if not condition:
        raise ValueError(message)


class Contract(BaseModel):
    """Reject unknown fields, coercions, and non-finite measurements."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Artifact(Contract):
    """An immutable input or retained output, without opening its location."""

    identifier: Text
    revision: Revision
    sha256: Sha256


class Hardware(Contract):
    """Hardware and software identity needed to interpret measurements."""

    platform: Text
    processor: Text
    device: Text
    threads: Positive
    runtime_versions: Annotated[dict[Text, Text], Field(min_length=1)]

    @field_validator("device")
    @classmethod
    def canonical_cpu_device(cls, value: str) -> str:
        """Use one CPU identity for case variants and indexed CPU devices."""
        if re.fullmatch(r"cpu(?::[0-9]+)?", value.strip(), flags=re.IGNORECASE):
            return "cpu"
        return value


class Protocol(Contract):
    """Shared comparison choices frozen before evaluation."""

    schema_version: Literal["1.0"]
    task: Task
    code_revision: Commit
    dependency_lock_sha256: Sha256
    dataset: Artifact
    annotation_quality: Literal["human_gold", "silver"]
    split: Literal["development", "test"]
    stage: Literal["screening", "final"]
    threshold_selection: Literal["development", "fixed_before_evaluation"]
    selection_sha256: Sha256 | None
    hardware: Hardware
    timing_policy: Literal["separate_fetch_load_warmup_steady"]
    memory_policy: Literal["process_peak_rss_and_device_peak_bytes"]
    invalid_output_policy: Literal["count_as_false_positive_and_retain"]
    uncertainty: Literal["marginal_stratified_document_bootstrap_95_percent"]
    bootstrap_seed: Count
    bootstrap_resamples: Annotated[int, Field(ge=1000)]

    @model_validator(mode="after")
    def selection_policy(self) -> Protocol:
        """Development screens candidates; tests only evaluate frozen finalists."""
        expected_split = "test" if self.stage == "final" else "development"
        require(self.split == expected_split, "selection stage and split disagree")
        require(
            self.stage != "final" or self.selection_sha256 is not None,
            "Final selection needs a frozen development decision artifact",
        )
        return self


class ReviewedCode(Contract):
    """Custom model code with a retained review for the exact code artifact."""

    code: Artifact
    review: Artifact


class Configuration(Contract):
    """One pipeline, source slice, language and seed in the planned inventory."""

    key: Text
    pipeline: Text
    language: Text
    source_config: Text
    examples: Positive
    gold_spans: Count
    sample_sha256: Sha256
    models: Annotated[list[Artifact], Field(min_length=1)]
    label_mapping: Annotated[dict[Text, Literal["LOC", "ignore"]], Field(min_length=1)]
    thresholds: dict[Text, float | None]
    batch_size: Positive
    seed: Count
    language_support: Literal[
        "documented", "transfer", "unspecified", "out_of_language_control"
    ]
    training_overlap: Literal["known", "unknown", "verified_absent"]
    provenance_note: Text
    gazetteer: Artifact | None
    parameters: dict[Text, JsonValue] = Field(default_factory=dict)
    custom_code: list[ReviewedCode] = Field(default_factory=list)

    @field_validator("custom_code")
    @classmethod
    def canonical_custom_code(cls, value: list[ReviewedCode]) -> list[ReviewedCode]:
        """Treat reviewed code as an unordered inventory, including in provenance."""
        entries = {entry.model_dump_json(): entry for entry in value}
        return [entries[key] for key in sorted(entries)]

    @field_validator("parameters")
    @classmethod
    def finite_parameters(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Reject nested NaN and infinity before hashing or serializing settings."""
        json.dumps(value, allow_nan=False)
        return value

    @model_validator(mode="after")
    def canonical_language(self) -> Configuration:
        """Reuse the authoritative 85-language inventory without normalizing scripts."""
        require(
            self.language in target_languages(), "language outside canonical inventory"
        )
        return self


class SpanCounts(Contract):
    """Exact, deduplicated, half-open character spans, with invalid outputs as FP."""

    gold_spans: Count
    predicted_spans: Count
    true_positive: Count
    false_positive: Count
    false_negative: Count
    invalid_outputs: Count

    @model_validator(mode="after")
    def denominators(self) -> SpanCounts:
        """Pin both count identities and the invalid-output penalty."""
        require(
            self.gold_spans == self.true_positive + self.false_negative,
            "gold span denominator disagrees with TP + FN",
        )
        require(
            self.predicted_spans == self.true_positive + self.false_positive,
            "predicted span denominator disagrees with TP + FP",
        )
        require(
            self.invalid_outputs <= self.false_positive,
            "invalid outputs must count as FP",
        )
        return self


class ResolutionCounts(Contract):
    """Gold-span resolution counts; every gold target has an explicit disposition."""

    task: Literal["gold_span_resolution"]
    gold_spans: Count
    resolved: Count
    abstained: Count
    invalid_outputs: Count
    exact_id_eligible: Count
    exact_id_correct: Count
    coordinate_eligible: Count
    within_1km: Count
    within_10km: Count
    within_50km: Count
    candidate_eligible: Count
    candidate_found: Count

    @model_validator(mode="after")
    def denominators(self) -> ResolutionCounts:
        """Check all-gold dispositions and each metric's eligible subset."""
        require(
            self.resolved + self.abstained + self.invalid_outputs == self.gold_spans,
            "resolution disposition denominator must equal gold spans",
        )
        require(
            self.exact_id_correct <= min(self.exact_id_eligible, self.resolved)
            and self.exact_id_eligible <= self.gold_spans,
            "exact ID denominator is inconsistent",
        )
        require(
            0
            <= self.within_1km
            <= self.within_10km
            <= self.within_50km
            <= min(self.coordinate_eligible, self.resolved)
            and self.coordinate_eligible <= self.gold_spans,
            "distance denominator or nested thresholds are inconsistent",
        )
        require(
            self.candidate_found <= self.candidate_eligible <= self.gold_spans,
            "candidate recall denominator is inconsistent",
        )
        require(
            self.candidate_eligible == self.exact_id_eligible,
            "candidate and exact-ID eligibility must use the same canonical gold IDs",
        )
        require(
            self.exact_id_correct <= self.candidate_found,
            "exact-ID successes cannot exceed retrieved candidate hits",
        )
        return self


class RecognitionCounts(SpanCounts):
    """A true positive requires an exact character span."""

    task: Literal["recognition"]


class EndToEndCounts(SpanCounts):
    """A true positive requires both the exact span and exact gazetteer ID."""

    task: Literal["end_to_end"]


Scores = Annotated[
    RecognitionCounts | ResolutionCounts | EndToEndCounts, Field(discriminator="task")
]


class Measurements(Contract):
    """Seconds and bytes, with data acquisition excluded from steady inference."""

    fetch_seconds: Measurement
    load_seconds: Measurement
    warmup_seconds: Measurement
    steady_seconds: Measurement
    warmup_examples: Count
    peak_rss_bytes: Positive
    peak_device_bytes: Count


class Unit(Contract):
    """One document or sentence, including failures, for aligned resampling."""

    example_id: Text
    failed: bool
    scores: Scores

    @model_validator(mode="after")
    def failure_policy(self) -> Unit:
        """A failed prediction can retain invalid outputs, but no valid prediction."""
        if not self.failed:
            return self
        if isinstance(self.scores, SpanCounts):
            require(
                self.scores.predicted_spans == self.scores.invalid_outputs,
                "failed unit cannot retain valid span predictions",
            )
        else:
            require(
                self.scores.resolved == 0,
                "failed unit cannot retain resolved predictions",
            )
        return self


class Result(Contract):
    """One explicit inventory outcome; a planned row is never a measured zero."""

    key: Text
    status: Literal["planned", "complete", "failed", "unsupported"]
    reason: Text | None
    provenance_sha256: Sha256 | None
    measurements: Measurements | None
    scores: Scores | None
    raw_predictions: Artifact | None
    evaluated_examples: Count
    failed_examples: Count
    units: list[Unit] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_payload(self) -> Result:
        """Separate completed measurements, failures, and unevaluated entries."""
        self._measurement_presence()
        if self.status in {"failed", "unsupported"}:
            require(
                self.reason is not None, "failed/unsupported result requires a reason"
            )
        if self.status in {"planned", "unsupported"}:
            self._unevaluated_payload()
        return self

    def _unevaluated_payload(self) -> None:
        """Keep execution evidence out of planned and unsupported inventory rows."""
        require(
            self.evaluated_examples + self.failed_examples == 0,
            "unevaluated result cannot claim example counts",
        )
        require(
            self.provenance_sha256 is None
            and self.raw_predictions is None
            and not self.units,
            "unevaluated result cannot supply provenance, raw predictions or unit scores",
        )

    def _measurement_presence(self) -> None:
        """Require all measurement components only for completed outcomes."""
        if self.status == "complete":
            values = (
                self.provenance_sha256,
                self.measurements,
                self.scores,
                self.raw_predictions,
            )
            require(
                all(value is not None for value in values),
                "complete result requires provenance, measurements, scores and raw predictions",
            )
        else:
            require(
                self.scores is None and self.measurements is None,
                "incomplete result cannot supply aggregate scores or measurements",
            )

    def completed_scores(self) -> Scores:
        """Return validated complete counts to callers that need non-optional data."""
        if self.scores is None:
            message = "Result has no complete scores"
            raise ValueError(message)
        return self.scores


class Experiment(Contract):
    """A complete inventory and outcomes under one comparable protocol."""

    protocol: Protocol
    configurations: Annotated[list[Configuration], Field(min_length=1)]
    results: list[Result]

    @model_validator(mode="after")
    def validate_inventory(self) -> Experiment:
        """Reject silent skips, duplicate cells, and altered source denominators."""
        keys = [config.key for config in self.configurations]
        result_keys = [result.key for result in self.results]
        require(len(set(keys)) == len(keys), "duplicate configuration in inventory")
        require(
            len(set(result_keys)) == len(result_keys) and set(keys) == set(result_keys),
            "result inventory must contain each configuration exactly once",
        )
        self._validate_sources()
        self._validate_pipeline_inventory()
        self._validate_results()
        return self

    def _validate_results(self) -> None:
        """Validate each outcome and the shared pairing of complete units."""
        configurations = {config.key: config for config in self.configurations}
        for result in self.results:
            self._validate_result(configurations[result.key], result)
        self._validate_pairing(configurations)

    def _validate_sources(self) -> None:
        """A model cannot change the examples or gold spans of a shared source."""
        sources: dict[tuple[str, str], tuple[int, int, str]] = {}
        cells: set[tuple[str, str, str, int]] = set()
        for config in self.configurations:
            cell = (config.pipeline, config.language, config.source_config, config.seed)
            require(cell not in cells, "duplicate pipeline/source/seed inventory cell")
            cells.add(cell)
            self._validate_thresholds(config)
            source = (config.language, config.source_config)
            identity = (config.examples, config.gold_spans, config.sample_sha256)
            require(
                sources.setdefault(source, identity) == identity,
                "mixed source sample or count denominators",
            )
            require(
                self.protocol.task == "recognition" or config.gazetteer is not None,
                "resolution and end-to-end require a pinned gazetteer",
            )

    def _validate_thresholds(self, config: Configuration) -> None:
        """Require explicit cutoffs and distinguish absent recognition cutoffs."""
        if self.protocol.task != "gold_span_resolution":
            require(
                "recognition" in config.thresholds,
                "recognition threshold is required; use null only when no cutoff applies",
            )
        if self.protocol.task != "recognition":
            require(
                config.thresholds.get("min_similarity") is not None,
                "resolution threshold min_similarity must be explicit and numeric",
            )

    def provenance_digest(self, config: Configuration) -> str:
        """Hash the canonical protocol and configuration, excluding result fields."""
        payload = {
            "protocol": self.protocol.model_dump(mode="json"),
            "configuration": config.model_dump(mode="json"),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _validate_result(self, config: Configuration, result: Result) -> None:
        """Link each result to its immutable inputs and declared sample size."""
        require(
            result.evaluated_examples + result.failed_examples <= config.examples,
            "example count exceeds source denominator",
        )
        if result.status != "complete":
            if result.status == "failed":
                self._validate_partial_evidence(config, result)
            return
        require(
            result.provenance_sha256 == self.provenance_digest(config),
            "mixed provenance: result digest differs from frozen configuration",
        )
        require(
            result.evaluated_examples + result.failed_examples == config.examples,
            "complete result must account for every source example",
        )
        if self.protocol.hardware.device == "cpu" and result.measurements is not None:
            require(
                result.measurements.peak_device_bytes == 0,
                "CPU runs must report zero device-memory bytes",
            )
        scores = result.completed_scores()
        require(scores.task == self.protocol.task, "result task differs from protocol")
        require(
            scores.gold_spans == config.gold_spans,
            "result gold span denominator changed",
        )
        self._validate_units(config, result)

    def _validate_partial_evidence(self, config: Configuration, result: Result) -> None:
        """Bind retained failed-run evidence to the frozen configuration."""
        if result.provenance_sha256 is not None:
            require(
                result.provenance_sha256 == self.provenance_digest(config),
                "mixed provenance: failed evidence differs from frozen configuration",
            )
        retained = bool(result.units) or result.raw_predictions is not None
        require(
            not retained or result.provenance_sha256 is not None,
            "retained failed-run evidence requires the configuration provenance digest",
        )
        self._validate_partial_units(config, result)

    def _validate_partial_units(self, config: Configuration, result: Result) -> None:
        """Keep failed-result evidence aligned with its retained source units."""
        ids = [unit.example_id for unit in result.units]
        require(
            len(result.units) <= config.examples,
            "unit inventory exceeds source example denominator",
        )
        require(len(ids) == len(set(ids)), "unit IDs must be unique")
        require(
            all(unit.scores.task == self.protocol.task for unit in result.units),
            "unit task differs from protocol",
        )
        require(
            len(result.units) == result.evaluated_examples + result.failed_examples,
            "unit count differs from reported processed examples",
        )
        require(
            sum(unit.failed for unit in result.units) == result.failed_examples,
            "unit failure count differs from reported failures",
        )
        require(
            sum(unit.scores.gold_spans for unit in result.units) <= config.gold_spans,
            "unit gold span total exceeds source denominator",
        )

    def _validate_units(self, config: Configuration, result: Result) -> None:
        """Retain one count record per source unit, including failed predictions."""
        ids = [unit.example_id for unit in result.units]
        require(
            len(ids) == len(set(ids)) == config.examples,
            "unit inventory must contain each source example exactly once",
        )
        require(
            sum(unit.failed for unit in result.units) == result.failed_examples,
            "unit failure count differs from reported failures",
        )
        require(
            all(unit.scores.task == self.protocol.task for unit in result.units),
            "unit task differs from protocol",
        )
        self._validate_unit_totals(result)

    def _validate_unit_totals(self, result: Result) -> None:
        """Compare every additive count with its reported total."""
        expected = result.completed_scores().model_dump(exclude={"task"})
        totals = {
            key: sum(getattr(unit.scores, key) for unit in result.units)
            for key in expected
        }
        require(totals == expected, "unit counts do not sum to result counts")

    def _validate_pipeline_inventory(self) -> None:
        """Keep model and reviewed code pins stable; retain unsupported cells."""
        groups: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
        identities: dict[str, tuple[list[Artifact], list[ReviewedCode]]] = {}
        for config in self.configurations:
            groups[(config.pipeline, config.seed)].add(
                (config.language, config.source_config)
            )
            identity = (config.models, config.custom_code)
            require(
                identities.setdefault(config.pipeline, identity) == identity,
                "pipeline mixes model or reviewed custom-code identities",
            )
        sources = set().union(*groups.values())
        require(
            all(group == sources for group in groups.values()),
            "pipeline inventory omits declared source configurations",
        )

    def _validate_pairing(self, configurations: dict[str, Configuration]) -> None:
        """Bind compared source samples to identical example IDs and gold counts."""
        observed: dict[tuple[str, str], dict[str, tuple[int, ...]]] = {}
        for result in self.results:
            if result.status == "complete":
                config = configurations[result.key]
                key = (config.language, config.source_config)
                gold = {
                    unit.example_id: _gold_identity(unit.scores)
                    for unit in result.units
                }
                require(
                    observed.setdefault(key, gold) == gold,
                    "paired source units or gold counts differ between pipelines",
                )


def _gold_identity(scores: Scores) -> tuple[int, ...]:
    """Keep eligible evaluation masks identical across compared predictions."""
    if isinstance(scores, ResolutionCounts):
        return (
            scores.gold_spans,
            scores.exact_id_eligible,
            scores.coordinate_eligible,
            scores.candidate_eligible,
        )
    return (scores.gold_spans,)
