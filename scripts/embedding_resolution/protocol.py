"""Freeze a gold-span resolution comparison as a validated protocol inventory.

A freeze plan holds only what an approved run can supply: the dataset samples,
the digest of each model's weights, development thresholds with their
calibration digests, and any reviewed custom code. This module expands the plan
into one planned configuration per pipeline and source, then lets the shared
benchmark-protocol validator judge the whole inventory. It never loads a model,
a dataset or a gazetteer, and every result it returns is ``planned``.
"""

from __future__ import annotations

from typing import Annotated, Literal, get_args

from pydantic import Field, model_validator

from geoparser.gazetteer.description import GAZETTEER_ATTRIBUTE_MAP
from scripts.benchmark_protocol.schema import (
    Artifact,
    Contract,
    Count,
    Experiment,
    Positive,
    Protocol,
    ReviewedCode,
    Sha256,
    Text,
    require,
)
from scripts.embedding_resolution.models import (
    MODELS,
    EmbeddingModel,
    get_model,
    historical_setting,
)
from scripts.embedding_resolution.resolution import (
    CALIBRATION_GRID,
    POPULATION_WEIGHT,
    Policy,
)

BASELINE_NAME = "baseline"
# The two policies every registered model is scored under, as the library names them.
EMBEDDING_POLICIES: tuple[str, ...] = get_args(Policy)
# GPE and LOC both name places, so both map to the protocol's single location label.
_DEFAULT_LABEL_MAPPING: dict[str, Literal["LOC", "ignore"]] = {
    "GPE": "LOC",
    "LOC": "LOC",
}
DESCRIPTION_FUNCTION = "geoparser.gazetteer.description.describe_feature"
PolicyName = Literal["similarity", "population", "population_only"]
Origin = Literal["development_calibrated", "historical", "structural"]


class SourceRecord(Contract):
    """One source slice shared by every pipeline, with its frozen sample digest."""

    language: Text
    source_config: Text
    examples: Positive
    gold_spans: Count
    sample_sha256: Sha256


class ContextPlan(Contract):
    """How reference contexts are cut, frozen identically for every model."""

    token_limit: Positive
    tokenizer: Artifact


class ThresholdRecord(Contract):
    """One pipeline's threshold, with the evidence that selected it."""

    model: Text | None
    policy: PolicyName
    min_similarity: float
    origin: Origin
    calibration_sha256: Sha256 | None
    note: Text

    @model_validator(mode="after")
    def provenance(self) -> ThresholdRecord:
        """Keep each threshold tied to the model and evidence that produced it."""
        if self.policy == "population_only":
            self._check_baseline()
        else:
            self._check_model_threshold()
        return self

    def _check_baseline(self) -> None:
        """The population-only baseline uses no model and no similarity cutoff."""
        require(self.model is None, "population-only baseline uses no embedding model")
        require(self.origin == "structural", "population-only threshold is structural")
        require(self.min_similarity == 0.0, "population-only baseline has no cutoff")
        require(self.calibration_sha256 is None, "structural threshold has no run")

    def _check_model_threshold(self) -> None:
        """An embedding threshold is calibrated for its model or labelled historical."""
        model = self.model
        if model is None:
            msg = "embedding pipeline needs a model key"
            raise ValueError(msg)
        require(
            model in {registered.key for registered in MODELS},
            "embedding pipeline names a model outside the registry",
        )
        require(self.origin != "structural", "only the baseline is structural")
        if self.origin == "development_calibrated":
            require(
                self.calibration_sha256 is not None,
                "a calibrated threshold needs its development calibration digest",
            )
            require(
                self.min_similarity in CALIBRATION_GRID,
                "a calibrated threshold must be a value on the development "
                "calibration grid",
            )
            return
        require(
            self.calibration_sha256 is None,
            "a historical setting has no development calibration digest",
        )
        require(
            historical_setting(model, self.min_similarity) is not None,
            "historical threshold is not registered for this model",
        )


class FreezePlan(Contract):
    """Everything an approved run must supply before the inventory can be frozen."""

    protocol: Protocol
    gazetteer: Artifact
    context: ContextPlan
    models: dict[Text, Artifact]
    reviewed_code: dict[Text, Annotated[list[ReviewedCode], Field(min_length=1)]] = (
        Field(default_factory=dict)
    )
    sources: Annotated[list[SourceRecord], Field(min_length=1)]
    thresholds: Annotated[list[ThresholdRecord], Field(min_length=1)]
    batch_size: Positive
    seed: Count
    label_mapping: dict[Text, Literal["LOC", "ignore"]] = Field(
        default_factory=lambda: dict(_DEFAULT_LABEL_MAPPING)
    )
    language_support: Literal[
        "documented", "transfer", "unspecified", "out_of_language_control"
    ] = "unspecified"
    training_overlap: Literal["known", "unknown", "verified_absent"] = "unknown"
    provenance_note: Text

    @model_validator(mode="after")
    def consistent_inventory(self) -> FreezePlan:
        """Reject a plan whose sources, thresholds or pins cannot be frozen."""
        require(
            self.protocol.task == "gold_span_resolution",
            "embedding comparison plans must use the gold_span_resolution task",
        )
        require(
            self.protocol.stage == "screening"
            and self.protocol.split == "development"
            and self.protocol.threshold_selection == "development",
            "embedding comparison plans are development screening only: stage "
            "screening, split development, development threshold selection",
        )
        require(
            self.gazetteer.identifier in GAZETTEER_ATTRIBUTE_MAP,
            "gazetteer has no attribute map for candidate descriptions",
        )
        self._check_sources_and_thresholds()
        self._check_complete()
        require(
            set(self.reviewed_code) <= {model.key for model in MODELS},
            "reviewed code names a model outside the registry",
        )
        return self

    def _check_sources_and_thresholds(self) -> None:
        """Each source and threshold appears once, and each model has a weight."""
        pairs = [(source.language, source.source_config) for source in self.sources]
        require(len(set(pairs)) == len(pairs), "duplicate source in freeze plan")
        identities = [
            (record.model, record.policy, record.min_similarity, record.origin)
            for record in self.thresholds
        ]
        require(len(set(identities)) == len(identities), "duplicate threshold record")

    def _check_complete(self) -> None:
        """Name every registered model under each embedding policy, and the baseline.

        Only development-calibrated rows and the structural baseline fill the
        comparison matrix, and each cell takes exactly one of them: a duplicate
        would run one cell twice, and a missing cell would never be run. A
        historical row is an extra. It never fills a cell, so a missing calibrated
        cell is refused even when a historical row exists for it. Each registered
        model also needs one weight artifact, and no other is accepted.
        """
        filling = [
            (record.model, record.policy)
            for record in self.thresholds
            if record.origin != "historical"
        ]
        require(
            len(set(filling)) == len(filling),
            "each model and policy needs exactly one development-calibrated "
            "threshold, and the baseline exactly once",
        )
        require(
            set(filling) == _expected_pairs(),
            "the plan must name every registered model under similarity and "
            "population, plus the population-only baseline",
        )
        require(
            set(self.models) == {model.key for model in MODELS},
            "the plan needs a weight artifact for each registered model and no other",
        )


def _expected_pairs() -> set[tuple[str | None, str]]:
    """Every registered model under each embedding policy, plus the baseline."""
    pairs: set[tuple[str | None, str]] = {(None, "population_only")}
    pairs.update(
        (model.key, policy) for model in MODELS for policy in EMBEDDING_POLICIES
    )
    return pairs


def _require_pin(model: EmbeddingModel, label: str, artifact: Artifact) -> None:
    """Require an artifact to be the registry's exact pin for the model."""
    if artifact.identifier != model.repository or artifact.revision != model.revision:
        msg = (
            f"{model.key}: {label} {artifact.identifier}@{artifact.revision} "
            f"differs from the registry pin {model.repository}@{model.revision}"
        )
        raise ValueError(msg)


def _pinned(plan: FreezePlan, model: EmbeddingModel) -> None:
    """Require the plan's weight artifact to be the registry's exact pin."""
    _require_pin(model, "plan artifact", plan.models[model.key])


def _reviewed(plan: FreezePlan, model: EmbeddingModel) -> list[ReviewedCode]:
    """
    Return the reviewed custom code for a model, refusing unreviewed or unpinned code.

    A review covers one code artifact, so each reviewed code artifact must be the
    registry's exact pin. A review of another repository or commit is refused.
    """
    reviewed = plan.reviewed_code.get(model.key, [])
    if model.trust_remote_code and not reviewed:
        msg = (
            f"{model.key} loads custom model code; it must be reviewed and pinned "
            "before the freeze can name it."
        )
        raise ValueError(msg)
    for entry in reviewed:
        _require_pin(model, "reviewed code", entry.code)
    return reviewed


def _parameters(
    plan: FreezePlan, record: ThresholdRecord, model: EmbeddingModel | None
) -> dict[str, object]:
    """The settings that enter the provenance digest of every configuration."""
    parameters: dict[str, object] = {
        "policy": record.policy,
        "population_weight": _weight(record),
        "threshold_origin": record.origin,
        "threshold_note": record.note,
        "calibration_sha256": record.calibration_sha256,
        "context_token_limit": plan.context.token_limit,
        "context_tokenizer": plan.context.tokenizer.model_dump(mode="json"),
        "candidate_description": DESCRIPTION_FUNCTION,
        "attribute_map": dict(GAZETTEER_ATTRIBUTE_MAP[plan.gazetteer.identifier]),
        "model": None,
    }
    if model is not None:
        parameters.update(_model_parameters(model))
    return parameters


def _weight(record: ThresholdRecord) -> float | None:
    """The population weight a policy applies; None where no model is scored."""
    if record.policy == "population_only":
        return None
    return POPULATION_WEIGHT if record.policy == "population" else 0.0


def _model_parameters(model: EmbeddingModel) -> dict[str, object]:
    """The pinned recipe for an embedding model, as protocol parameters."""
    return {
        "model": model.key,
        "repository": model.repository,
        "revision": model.revision,
        "dimension": model.dimension,
        "pooling": model.pooling,
        "normalize": model.normalize,
        "query_prompt": model.query_prompt,
        "document_prompt": model.document_prompt,
        "task": model.task,
        "max_seq_length": model.max_seq_length,
        "max_seq_length_source": model.max_seq_length_source,
        "trust_remote_code": model.trust_remote_code,
        "license": model.license,
    }


def _pipeline_key(record: ThresholdRecord) -> str:
    """A unique, readable pipeline name: model, policy, threshold and origin."""
    name = record.model or BASELINE_NAME
    return f"{name}-{record.policy}-t{record.min_similarity:+.2f}-{record.origin}"


def _configuration(
    plan: FreezePlan,
    record: ThresholdRecord,
    source: SourceRecord,
) -> dict[str, object]:
    """Build one planned configuration for a pipeline and a source slice."""
    if record.model is None:
        model, weight = None, plan.gazetteer
        reviewed: list[ReviewedCode] = []
    else:
        model = get_model(record.model)
        _pinned(plan, model)
        reviewed = _reviewed(plan, model)
        weight = plan.models[model.key]
    pipeline = _pipeline_key(record)
    return {
        "key": f"{pipeline}/{source.language}/{source.source_config}",
        "pipeline": pipeline,
        "language": source.language,
        "source_config": source.source_config,
        "examples": source.examples,
        "gold_spans": source.gold_spans,
        "sample_sha256": source.sample_sha256,
        "models": [weight.model_dump(mode="json")],
        "label_mapping": plan.label_mapping,
        "thresholds": {"min_similarity": record.min_similarity},
        "batch_size": plan.batch_size,
        "seed": plan.seed,
        "language_support": plan.language_support,
        "training_overlap": plan.training_overlap,
        "provenance_note": f"{plan.provenance_note} {record.note}",
        "gazetteer": plan.gazetteer.model_dump(mode="json"),
        "parameters": _parameters(plan, record, model),
        "custom_code": [code.model_dump(mode="json") for code in reviewed],
    }


def _planned_result(key: str) -> dict[str, object]:
    """A planned row: it names the cell and carries no execution evidence."""
    return {
        "key": key,
        "status": "planned",
        "reason": None,
        "provenance_sha256": None,
        "measurements": None,
        "scores": None,
        "raw_predictions": None,
        "evaluated_examples": 0,
        "failed_examples": 0,
        "units": [],
    }


def build_experiment(plan: FreezePlan) -> Experiment:
    """
    Expand a freeze plan into a validated, fully planned inventory.

    Every threshold record becomes one pipeline, and every pipeline covers every
    source slice, so no comparison cell can be silently omitted.

    Args:
        plan: The validated freeze plan

    Returns:
        The validated experiment, with one planned result per configuration

    Raises:
        ValueError: If a model is unpinned, a custom-code model has no review, or
            the inventory violates the benchmark protocol
    """
    configurations = [
        _configuration(plan, record, source)
        for record in plan.thresholds
        for source in plan.sources
    ]
    results = [_planned_result(str(config["key"])) for config in configurations]
    return Experiment.model_validate(
        {
            "protocol": plan.protocol.model_dump(mode="json"),
            "configurations": configurations,
            "results": results,
        }
    )
