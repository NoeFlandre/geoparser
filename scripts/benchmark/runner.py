"""
Run a pipeline over one corpus, checkpointing as it goes.

Documents are processed in chunks and each chunk is persisted before the next
begins, so a job that runs out of walltime loses at most one chunk. The unit
of resumption is the document, because that is the unit the scoring is defined
over: half a document's toponyms would be a wrong number, not a partial one.
"""

from __future__ import annotations

import gc
import json
import time
import typing as t
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from geoparser.modules.recognizers.otter import PredictionOutputError
from scripts._io import write_text_atomic
from scripts.benchmark import checkpoint as ckpt
from scripts.benchmark import pipelines, report
from scripts.benchmark.corpus import Document

RECOGNITION = "recognition"
RESOLUTION = "resolution"
DEFAULT_CHUNK_SIZE = 5


@dataclass(frozen=True)
class _PhaseExecution:
    """Inputs shared by all chunk operations in one benchmark phase."""

    phase: str
    documents: Sequence[Document]
    state: ckpt.Checkpoint
    checkpoint_path: Path
    device: str
    chunk_size: int
    log: t.Callable[[str], None]


@dataclass(frozen=True)
class PhaseSettings:
    """How one benchmark phase runs: where, at what threshold, and how it reports."""

    device: str
    min_similarity: float
    chunk_size: int
    log: t.Callable[[str], None] = print


def gold_annotations(documents: Sequence[Document]) -> list[t.Any]:
    """Return every gold span as an annotation carrying its coordinates."""
    from geoparser.evaluation import Annotation

    return [
        Annotation(
            span.start,
            span.end,
            None,
            document.identifier,
            span.latitude,
            span.longitude,
        )
        for document in documents
        for span in document.gold
    ]


def scoring_span_count(documents: Sequence[Document]) -> int:
    """Count distinct document/span identities used by the set-based metrics."""
    return len(
        {
            (document.identifier, span.start, span.end)
            for document in documents
            for span in document.gold
        }
    )


def inspection_records(
    documents: Sequence[Document],
    state: ckpt.Checkpoint,
    *,
    pipeline: str,
    scoring_view: str,
) -> list[dict[str, t.Any]]:
    """Return one text-free reference/prediction record per input document."""
    records = []
    for document in documents:
        failures = {
            phase: state.failures.get(phase, {}).get(document.identifier)
            for phase in (RECOGNITION, RESOLUTION)
        }
        phase_status = {
            phase: (
                "failed"
                if failures[phase]
                else "complete"
                if document.identifier in state.completed(phase)
                else "incomplete"
            )
            for phase in (RECOGNITION, RESOLUTION)
        }
        safe_failures = {
            phase: {key: failure[key] for key in ("kind", "reason") if key in failure}
            for phase, failure in failures.items()
            if failure
        }
        records.append(
            {
                "schema": "geoparser-prediction-inspection-v1",
                "pipeline": pipeline,
                "scoring_view": scoring_view,
                "document_id": document.identifier,
                "reference": [
                    {
                        "start": span.start,
                        "end": span.end,
                        "name": span.name,
                        "latitude": span.latitude,
                        "longitude": span.longitude,
                    }
                    for span in document.gold
                ],
                "resolution_input": "gold_reference",
                "prediction": {
                    phase: [
                        {
                            key: value
                            for key, value in annotation.items()
                            if key != "document_id"
                        }
                        for annotation in state._phase(phase).get(
                            document.identifier, []
                        )
                    ]
                    for phase in (RECOGNITION, RESOLUTION)
                },
                "phase_status": phase_status,
                "failures": safe_failures,
            }
        )
    return records


def write_inspection(
    path: Path,
    documents: Sequence[Document],
    state: ckpt.Checkpoint,
    *,
    pipeline: str,
    scoring_view: str,
) -> None:
    """Write complete text-free per-document references and predictions."""
    lines = [
        json.dumps(record, ensure_ascii=False, allow_nan=False)
        for record in inspection_records(
            documents, state, pipeline=pipeline, scoring_view=scoring_view
        )
    ]
    write_text_atomic(path, "\n".join(lines) + ("\n" if lines else ""))


def predictions_by_document(
    project: t.Any, document_ids: Sequence[t.Any], documents: Sequence[Document]
) -> dict[str, list[dict[str, t.Any]]]:
    """
    Read predictions back, keyed by the document they belong to.

    Documents are asked for by the identifiers ``create_documents`` returned,
    because an unfiltered read has no ordering contract.

    Args:
        project: The project the documents were created in
        document_ids: The identifiers create_documents returned
        documents: The corpus documents, in the same order

    Returns:
        Corpus document identifier to serialized annotations
    """
    from geoparser.evaluation import toponym_annotation

    predictions: dict[str, list[dict[str, t.Any]]] = {}
    for document, parsed in zip(
        documents, project.get_documents(list(document_ids)), strict=True
    ):
        annotations = [
            report.serialize(
                toponym_annotation(toponym, document.identifier, with_coordinates=True)
            )
            for toponym in parsed.toponyms
        ]
        predictions[document.identifier] = annotations
    return predictions


def chunks(items: Sequence[t.Any], size: int) -> list[list[t.Any]]:
    """
    Split a sequence into consecutive chunks of at most ``size``.

    Args:
        items: What to split
        size: Maximum chunk length, at least one

    Returns:
        The chunks, in order

    Raises:
        ValueError: If the size is not positive, which would never terminate
    """
    if size < 1:
        msg = f"chunk size must be at least 1, got {size}"
        raise ValueError(msg)
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def run_phase(
    phase: str,
    pipeline: str,
    documents: Sequence[Document],
    state: ckpt.Checkpoint,
    checkpoint_path: Path,
    settings: PhaseSettings,
) -> dict[str, str]:
    """
    Run one phase over every document not already checkpointed.

    Args:
        phase: ``recognition`` or ``resolution``
        pipeline: A name registered in ``scripts.benchmark.pipelines``
        documents: The whole corpus slice under test
        state: The checkpoint to read from and extend
        checkpoint_path: Where to persist the checkpoint
        settings: Device, shared abstention threshold for resolution, chunk
            size between saves, and where progress goes

    Returns:
        The checkpoints the models loaded, for the report
    """
    done = state.completed(phase)
    remaining = [d for d in documents if d.identifier not in done]
    if not remaining:
        settings.log(f"  {pipeline}/{phase}: already complete ({len(done)} documents)")
        return {}

    settings.log(
        f"  {pipeline}/{phase}: {len(remaining)} of {len(documents)} documents "
        f"to do on {settings.device}"
    )

    # Model construction is inside the timer: loading a cross encoder is a
    # real cost of choosing that pipeline, and leaving it out understated the
    # slower one by minutes.
    loading_started = time.perf_counter()
    recognizer, resolver = _build_phase_models(
        phase,
        pipeline,
        device=settings.device,
        min_similarity=settings.min_similarity,
    )
    names = pipelines.model_names(recognizer, resolver)
    loading_seconds = time.perf_counter() - loading_started
    state.elapsed_seconds += loading_seconds
    state.phase_elapsed_seconds[phase] = (
        state.phase_elapsed_seconds.get(phase, 0.0) + loading_seconds
    )
    _run_chunks(
        _PhaseExecution(
            phase,
            documents,
            state,
            checkpoint_path,
            settings.device,
            settings.chunk_size,
            settings.log,
        ),
        remaining,
        (recognizer, resolver),
    )
    return names


def _build_phase_models(
    phase: str, pipeline: str, *, device: str, min_similarity: float
) -> tuple[t.Any, t.Any]:
    """Build only the model required by the requested benchmark phase."""
    if phase == RECOGNITION:
        return pipelines.build_recognizer(pipeline, device=device), None
    return (
        None,
        pipelines.build_resolver(
            pipeline, device=device, min_similarity=min_similarity
        ),
    )


def _run_chunks(
    execution: _PhaseExecution,
    remaining: Sequence[Document],
    models: tuple[t.Any, t.Any],
) -> None:
    """Process and checkpoint each chunk, then release the phase models."""
    phase_chunks = chunks(remaining, execution.chunk_size)
    recognizer, resolver = models

    try:
        for index, chunk in enumerate(phase_chunks, start=1):
            started = time.perf_counter()
            predictions, failures = _process_chunk_with_failures(
                execution.phase,
                chunk,
                recognizer=recognizer,
                resolver=resolver,
            )
            for document_id, annotations in predictions.items():
                execution.state.record(execution.phase, document_id, annotations)
            if failures:
                execution.state.failures.setdefault(execution.phase, {}).update(
                    failures
                )
            elapsed = time.perf_counter() - started
            execution.state.elapsed_seconds += elapsed
            execution.state.phase_elapsed_seconds[execution.phase] = (
                execution.state.phase_elapsed_seconds.get(execution.phase, 0.0)
                + elapsed
            )
            ckpt.save(execution.checkpoint_path, execution.state)
            execution.log(
                f"    chunk {index}/{len(phase_chunks)} "
                f"({len(chunk)} docs) saved; "
                f"{len(execution.state.completed(execution.phase))}/"
                f"{len(execution.documents)} done"
            )
    finally:
        del recognizer, resolver
        gc.collect()
        _empty_cuda_cache(execution.device)


def _empty_cuda_cache(device: str) -> None:
    """Release cached GPU memory so the next phase starts with a clean device."""
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.empty_cache()


def _process_chunk(
    phase: str,
    documents: Sequence[Document],
    *,
    recognizer: t.Any,
    resolver: t.Any,
) -> dict[str, list[dict[str, t.Any]]]:
    """
    Run one chunk through a fresh project and return its predictions.

    A project per chunk keeps the database small and means an interrupted job
    leaves nothing to clean up beyond the chunk it was working on.
    """
    from geoparser.modules.recognizers.manual import ManualRecognizer
    from geoparser.project import Project

    texts = [document.text for document in documents]
    project = Project(f"bench-{uuid.uuid4().hex[:8]}")
    try:
        document_ids = project.create_documents(texts)
        if phase == RECOGNITION:
            project.run_recognizer(recognizer)
        else:
            references = [
                [(span.start, span.end) for span in document.gold]
                for document in documents
            ]
            project.run_recognizer(ManualRecognizer("gold", texts, references))
            project.run_resolver(resolver)
        return predictions_by_document(project, document_ids, documents)
    finally:
        project.delete()


def _process_chunk_with_failures(
    phase: str,
    documents: Sequence[Document],
    *,
    recognizer: t.Any,
    resolver: t.Any,
) -> tuple[dict[str, list[dict[str, t.Any]]], dict[str, dict[str, t.Any]]]:
    """Record Otter's document-level input and output failures explicitly."""
    try:
        predictions = _process_chunk(
            phase, documents, recognizer=recognizer, resolver=resolver
        )
    except ValueError as error:
        if phase != RECOGNITION or _otter_failure(error) is None:
            raise
        return _process_recognition_documents_individually(
            documents, recognizer=recognizer, resolver=resolver
        )

    return predictions, {}


def _process_recognition_documents_individually(
    documents: Sequence[Document], *, recognizer: t.Any, resolver: t.Any
) -> tuple[dict[str, list[dict[str, t.Any]]], dict[str, dict[str, t.Any]]]:
    """Retry a failed Otter batch per document and retain its failures."""
    predictions: dict[str, list[dict[str, t.Any]]] = {}
    failures: dict[str, dict[str, t.Any]] = {}
    for document in documents:
        try:
            predictions.update(
                _process_chunk(
                    RECOGNITION,
                    [document],
                    recognizer=recognizer,
                    resolver=resolver,
                )
            )
        except ValueError as error:
            failure = _otter_failure(error)
            if failure is None:
                raise
            predictions[document.identifier] = []
            failures[document.identifier] = {
                "kind": failure[0],
                "reason": failure[1],
                "message": str(error),
            }
            if isinstance(error, PredictionOutputError):
                failures[document.identifier]["raw_output"] = _json_safe_raw_output(
                    error.raw_output
                )
    return predictions, failures


_MAX_RAW_OUTPUT_JSON_CHARS = 4096


def _json_safe_raw_output(value: t.Any) -> t.Any:
    """Bound and normalize diagnostic model output before checkpointing it."""
    if _json_value_size(value, _MAX_RAW_OUTPUT_JSON_CHARS, 0, set()) is None:
        return {
            "omitted": True,
            "reason": "non_json_or_over_character_limit",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    return json.loads(json.dumps(value, allow_nan=False))


def _json_value_size(
    value: t.Any,
    budget: int,
    depth: int,
    ancestors: set[int],
) -> int | None:
    """Measure strict JSON-compatible values without copying them."""
    if _json_size_budget_exceeded(budget, depth):
        return None
    return _json_size_for_type(value, budget, depth, ancestors)


def _json_size_budget_exceeded(budget: int, depth: int) -> bool:
    """Whether a value has exhausted its character or nesting budget."""
    return budget < 1 or depth > 32


def _json_size_for_type(
    value: t.Any,
    budget: int,
    depth: int,
    ancestors: set[int],
) -> int | None:
    """Dispatch a bounded JSON value by its exact built-in type."""
    if type(value) in {type(None), bool, int, float}:
        return _json_primitive_size(value, budget)
    if type(value) is str:
        return _json_string_size(value, budget)
    return _json_container_size(value, budget, depth, ancestors)


def _json_container_size(
    value: t.Any,
    budget: int,
    depth: int,
    ancestors: set[int],
) -> int | None:
    """Dispatch supported JSON containers without accepting subclasses."""
    if type(value) in {list, tuple}:
        return _json_sequence_size(value, budget, depth, ancestors)
    if type(value) is dict:
        return _json_mapping_size(value, budget, depth, ancestors)
    return None


def _json_primitive_size(value: t.Any, budget: int) -> int | None:
    """Measure a JSON primitive after checking its bounded scalar encoding."""
    if type(value) is int and value.bit_length() > budget * 4:
        return None
    try:
        size = len(json.dumps(value, allow_nan=False))
    except (OverflowError, ValueError):
        return None
    return size if size <= budget else None


def _json_string_size(value: str, budget: int) -> int | None:
    """Measure an ASCII-safe JSON string, stopping as soon as it is too large."""
    size = 2
    if budget < size:
        return None
    for character in value:
        size += _json_character_size(character)
        if size > budget:
            return None
    return size


def _json_character_size(character: str) -> int:
    """Return a conservative ASCII JSON-escape length for one character."""
    if character in {'"', "\\"}:
        return 2
    codepoint = ord(character)
    if codepoint < 0x20 or codepoint >= 0x7F:
        return 6 if codepoint <= 0xFFFF else 12
    return 1


def _json_sequence_size(
    values: list[t.Any] | tuple[t.Any, ...],
    budget: int,
    depth: int,
    ancestors: set[int],
) -> int | None:
    """Measure list/tuple JSON payloads with bounded traversal."""
    identity = id(values)
    if identity in ancestors or budget < 2:
        return None
    ancestors.add(identity)
    try:
        return _json_sequence_content_size(values, budget, depth, ancestors)
    finally:
        ancestors.remove(identity)


def _json_sequence_content_size(
    values: list[t.Any] | tuple[t.Any, ...],
    budget: int,
    depth: int,
    ancestors: set[int],
) -> int | None:
    """Measure sequence items, including the standard JSON separators."""
    size = 2
    for index, value in enumerate(values):
        if index:
            size += 2
        item_size = _json_value_size(value, budget - size, depth + 1, ancestors)
        if item_size is None:
            return None
        size += item_size
    return size if size <= budget else None


def _json_mapping_size(
    values: dict[str, t.Any],
    budget: int,
    depth: int,
    ancestors: set[int],
) -> int | None:
    """Measure string-keyed JSON objects with bounded traversal."""
    identity = id(values)
    if identity in ancestors or budget < 2:
        return None
    ancestors.add(identity)
    try:
        return _json_mapping_content_size(values, budget, depth, ancestors)
    finally:
        ancestors.remove(identity)


def _json_mapping_content_size(
    values: dict[str, t.Any],
    budget: int,
    depth: int,
    ancestors: set[int],
) -> int | None:
    """Measure mapping entries, including default JSON spacing."""
    size = 2
    for index, (key, value) in enumerate(values.items()):
        if index:
            size += 2
        entry_size = _json_mapping_entry_size(
            key, value, budget - size, depth, ancestors
        )
        if entry_size is None:
            return None
        size += entry_size
    return size if size <= budget else None


def _json_mapping_entry_size(
    key: t.Any,
    value: t.Any,
    budget: int,
    depth: int,
    ancestors: set[int],
) -> int | None:
    """Measure one string-keyed mapping entry without copying its value."""
    if type(key) is not str:
        return None
    key_size = _json_string_size(key, budget)
    if key_size is None:
        return None
    value_size = _json_value_size(value, budget - key_size - 2, depth + 1, ancestors)
    if value_size is None:
        return None
    return key_size + 2 + value_size


def _otter_failure(error: ValueError) -> tuple[str, str] | None:
    """Return safe report metadata for the two document-level Otter failures."""
    message = str(error)
    if "Otter document at index" in message and "exceeding the limit" in message:
        return (
            "unsupported_input",
            "Otter sequence-length limit; document was not truncated",
        )
    if isinstance(error, PredictionOutputError):
        return (
            "invalid_output",
            "Otter returned output that did not match its document.",
        )
    return None


def score(
    pipeline: str,
    documents: Sequence[Document],
    state: ckpt.Checkpoint,
    *,
    models: dict[str, str],
    device: str,
) -> report.PipelineResult:
    """
    Score a completed checkpoint.

    Args:
        pipeline: Which pipeline the state belongs to
        documents: The corpus slice under test
        state: The checkpoint holding its predictions
        models: The checkpoints the models loaded
        device: Where the run happened, for the report

    Returns:
        The scored result
    """
    expected = gold_annotations(documents)
    result = report.PipelineResult(
        name=pipeline,
        models=models,
        elapsed_seconds=state.elapsed_seconds,
        phase_elapsed_seconds=state.phase_elapsed_seconds.copy(),
        device=device,
        phase_provenance=state.phase_provenance.copy(),
        failures=state.failures.copy(),
    )
    recognized = [report.deserialize(a) for a in state.annotations(RECOGNITION)]
    resolved = [report.deserialize(a) for a in state.annotations(RESOLUTION)]
    complete_documents = {document.identifier for document in documents}
    result.recognition = _score_completed_phase(
        RECOGNITION,
        expected,
        recognized,
        complete_documents,
        state,
        invalid_output_false_positives=sum(
            failure.get("kind") == "invalid_output"
            for failure in state.failures.get(RECOGNITION, {}).values()
        ),
    )
    result.resolution = _score_completed_phase(
        RESOLUTION, expected, resolved, complete_documents, state
    )
    return result


def _score_completed_phase(
    phase: str,
    expected: Sequence[report.Annotation],
    predictions: Sequence[report.Annotation],
    document_ids: set[str],
    state: ckpt.Checkpoint,
    *,
    invalid_output_false_positives: int = 0,
) -> dict[str, float]:
    """Return metrics only when every document has a prediction for the phase."""
    if state.completed(phase) != document_ids:
        return {}
    if phase == RECOGNITION:
        return report.score_recognition(
            expected,
            predictions,
            invalid_output_false_positives=invalid_output_false_positives,
        )
    return report.score_resolution(expected, predictions)
