"""
Score a benchmark run and render its report.

Kept free of models and the gazetteer so the scoring can be tested against
hand-written annotations, which is the part that must not be wrong: a bug here
would not crash, it would produce a plausible number.
"""

from __future__ import annotations

import typing as t
from collections.abc import Sequence
from dataclasses import dataclass, field

from geoparser.evaluation import (
    Annotation,
    accuracy_at_km,
    area_under_error_curve,
    mean_error_km,
    median_error_km,
    recognition_f1,
    recognition_precision,
    recognition_recall,
)


@dataclass
class PipelineResult:
    """What one pipeline scored, how long it took, and on what."""

    name: str
    models: dict[str, str] = field(default_factory=dict)
    recognition: dict[str, float] = field(default_factory=dict)
    resolution: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    device: str = "cpu"


def serialize(annotation: Annotation) -> dict[str, t.Any]:
    """Return an annotation as plain JSON-safe data."""
    return {
        "start": annotation.start,
        "end": annotation.end,
        "identifier": annotation.identifier,
        "document_id": annotation.document_id,
        "latitude": annotation.latitude,
        "longitude": annotation.longitude,
    }


def deserialize(payload: dict[str, t.Any]) -> Annotation:
    """Rebuild an annotation from plain data."""
    return Annotation(
        payload["start"],
        payload["end"],
        payload.get("identifier"),
        payload.get("document_id"),
        payload.get("latitude"),
        payload.get("longitude"),
    )


def score_recognition(
    expected: Sequence[Annotation], predicted: Sequence[Annotation]
) -> dict[str, float]:
    """Return precision, recall and F1 over exactly matched spans."""
    return {
        "precision": recognition_precision(expected, predicted),
        "recall": recognition_recall(expected, predicted),
        "f1": recognition_f1(expected, predicted),
    }


def score_resolution(
    expected: Sequence[Annotation], predicted: Sequence[Annotation]
) -> dict[str, float]:
    """Return the distance metrics for a set of placed toponyms."""
    return {
        "accuracy_at_161km": accuracy_at_km(expected, predicted),
        "mean_error_km": mean_error_km(expected, predicted),
        "median_error_km": median_error_km(expected, predicted),
        "auc": area_under_error_curve(expected, predicted),
    }


def render_markdown(
    results: Sequence[PipelineResult],
    *,
    corpus_name: str,
    documents: int,
    gold_toponyms: int,
    gazetteer: str,
    min_similarity: float,
) -> str:
    """
    Render the comparison as a report.

    Args:
        results: One entry per pipeline that ran
        corpus_name: Which corpus the numbers come from
        documents: How many documents were scored
        gold_toponyms: How many gold toponyms were scored
        gazetteer: The gazetteer both pipelines resolved against
        min_similarity: The shared abstention threshold

    Returns:
        The report as Markdown
    """
    lines = [
        f"# {corpus_name} benchmark",
        "",
        f"- Documents: {documents}",
        f"- Gold toponyms: {gold_toponyms}",
        f"- Gazetteer: {gazetteer}",
        f"- Shared abstention threshold: {min_similarity}",
        "",
        "## Recognition (each pipeline's own spans, exact match)",
        "",
        "| Pipeline | Device | Precision | Recall | F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {result.name} | {result.device} "
        f"| {result.recognition['precision']:.3f} "
        f"| {result.recognition['recall']:.3f} "
        f"| {result.recognition['f1']:.3f} |"
        for result in results
        if result.recognition
    ]
    lines += [
        "",
        "## Resolution (gold spans supplied, distance scored)",
        "",
        "| Pipeline | Device | Acc@161km | Mean err (km) | Median err (km) "
        "| AUC | Seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {result.name} | {result.device} "
        f"| {result.resolution['accuracy_at_161km']:.3f} "
        f"| {result.resolution['mean_error_km']:.1f} "
        f"| {result.resolution['median_error_km']:.1f} "
        f"| {result.resolution['auc']:.3f} "
        f"| {result.elapsed_seconds:.1f} |"
        for result in results
        if result.resolution
    ]
    lines += [
        "",
        "## Models",
        "",
    ]
    for result in results:
        models = ", ".join(f"{role}={name}" for role, name in result.models.items())
        lines.append(f"- **{result.name}**: {models}")
    lines += [
        "",
        "Resolution is scored on the gold spans, supplied to both pipelines",
        "through ManualRecognizer, so the resolvers are judged on the same",
        "toponyms and a difference cannot be an artefact of recognition.",
        "",
        "A gold toponym a pipeline did not place is charged the maximum",
        "possible error rather than dropped, so resolving less cannot improve",
        "a score. The shared abstention threshold is stated above. Models",
        "can have different similarity scales, so the threshold is part of",
        "the experiment and should be considered when comparing resolvers.",
        "",
        "AUC is a log-scaled summary of the whole error distribution, lower",
        "being better. Its normalization is this harness's own, so compare",
        "runs of this harness rather than published AUC figures.",
        "",
    ]
    return "\n".join(lines)


def render_summary(rows: Sequence[dict[str, t.Any]]) -> str:
    """
    Render one table across every corpus and pipeline that ran.

    Args:
        rows: One entry per corpus and pipeline, as written to summary.json

    Returns:
        The summary as Markdown
    """
    lines = [
        "# Benchmark summary",
        "",
        "Recognition is exact-match F1 on each pipeline's own spans; resolution",
        "is scored on the gold spans. Each corpus's own report has the detail.",
        "",
        "| Corpus | Lang | Docs | Gold | Pipeline | Rec F1 | Acc@161km | AUC "
        "| Seconds |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        f1 = row["recognition"].get("f1") if row["recognition"] else None
        resolution = row["resolution"] or {}
        lines.append(
            f"| {row['corpus']} | {row['language']} | {row['documents']} "
            f"| {row['gold_toponyms']} | {row['pipeline']} "
            f"| {_number(f1)} | {_number(resolution.get('accuracy_at_161km'))} "
            f"| {_number(resolution.get('auc'))} | {row['elapsed_seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def _number(value: float | None) -> str:
    """Format a score, or a dash for a phase that did not run."""
    return "-" if value is None else f"{value:.3f}"
