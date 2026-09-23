"""
Run a pipeline over the corpus, checkpointing as it goes.

Documents are processed in chunks and each chunk is persisted before the next
begins, so a job that runs out of walltime loses at most one chunk. The unit
of resumption is the document, because that is the unit the scoring is defined
over: half a document's toponyms would be a wrong number, not a partial one.
"""

from __future__ import annotations

import gc
import time
import typing as t
import uuid
from collections.abc import Sequence
from pathlib import Path

from scripts.benchmark import checkpoint as ckpt
from scripts.benchmark import pipelines, report
from scripts.benchmark.corpus import Document

RECOGNITION = "recognition"
RESOLUTION = "resolution"


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


def _coordinates(location: t.Any) -> tuple[float | None, float | None]:
    """Return a resolved location's coordinates, when it has usable ones."""
    if location is None:
        return None, None
    data = getattr(location, "data", None) or {}
    try:
        return float(data["latitude"]), float(data["longitude"])
    except (KeyError, TypeError, ValueError):
        return None, None


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
    from geoparser.evaluation import Annotation

    predictions: dict[str, list[dict[str, t.Any]]] = {}
    for document, parsed in zip(
        documents, project.get_documents(list(document_ids)), strict=True
    ):
        annotations = []
        for toponym in parsed.toponyms:
            latitude, longitude = _coordinates(toponym.location)
            annotations.append(
                report.serialize(
                    Annotation(
                        toponym.start,
                        toponym.end,
                        getattr(toponym.location, "identifier", None),
                        document.identifier,
                        latitude,
                        longitude,
                    )
                )
            )
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
        raise ValueError(f"chunk size must be at least 1, got {size}")
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def run_phase(
    phase: str,
    pipeline: str,
    documents: Sequence[Document],
    state: ckpt.Checkpoint,
    checkpoint_path: Path,
    *,
    device: str,
    min_similarity: float,
    chunk_size: int,
    log: t.Callable[[str], None] = print,
) -> dict[str, str]:
    """
    Run one phase over every document not already checkpointed.

    Args:
        phase: ``recognition`` or ``resolution``
        pipeline: ``upstream`` or ``swapped``
        documents: The whole corpus slice under test
        state: The checkpoint to read from and extend
        checkpoint_path: Where to persist the checkpoint
        device: Where to place the models
        min_similarity: Shared abstention threshold, for resolution
        chunk_size: How many documents to process between saves
        log: Where progress goes

    Returns:
        The checkpoints the models loaded, for the report
    """
    done = state.completed(phase)
    remaining = [d for d in documents if d.identifier not in done]
    if not remaining:
        log(f"  {pipeline}/{phase}: already complete ({len(done)} documents)")
        return {}

    log(
        f"  {pipeline}/{phase}: {len(remaining)} of {len(documents)} documents "
        f"to do on {device}"
    )

    # Model construction is inside the timer: loading a cross encoder is a
    # real cost of choosing that pipeline, and leaving it out understated the
    # slower one by minutes.
    loading_started = time.perf_counter()
    recognizer = resolver = None
    if phase == RECOGNITION:
        recognizer = pipelines.build_recognizer(pipeline, device=device)
    else:
        resolver = pipelines.build_resolver(
            pipeline, device=device, min_similarity=min_similarity
        )
    names = pipelines.model_names(recognizer, resolver)
    state.elapsed_seconds += time.perf_counter() - loading_started

    try:
        for index, chunk in enumerate(chunks(remaining, chunk_size), start=1):
            started = time.perf_counter()
            predictions = _process_chunk(
                phase, chunk, recognizer=recognizer, resolver=resolver
            )
            for document_id, annotations in predictions.items():
                state.record(phase, document_id, annotations)
            state.elapsed_seconds += time.perf_counter() - started
            ckpt.save(checkpoint_path, state)
            log(
                f"    chunk {index}/{len(chunks(remaining, chunk_size))} "
                f"({len(chunk)} docs) saved; "
                f"{len(state.completed(phase))}/{len(documents)} done"
            )
    finally:
        del recognizer, resolver
        gc.collect()
        _empty_cuda_cache(device)

    return names


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
        device=device,
    )
    recognized = [report.deserialize(a) for a in state.annotations(RECOGNITION)]
    resolved = [report.deserialize(a) for a in state.annotations(RESOLUTION)]
    if recognized:
        result.recognition = report.score_recognition(expected, recognized)
    if resolved:
        result.resolution = report.score_resolution(expected, resolved)
    return result
