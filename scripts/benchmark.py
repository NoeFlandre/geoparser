"""
Run a public geoparsing benchmark against two pipelines and compare them.

The corpus is GeoVirus (Gritta et al., *A Pragmatic Guide to Geoparsing
Evaluation*): 239 WikiNews articles whose gold toponyms carry coordinates but
no gazetteer identifiers. That is why this harness scores distance rather than
identifier equality -- it is the only thing two different gazetteers, or a
gazetteer and a Wikipedia page, can be compared on.

Recognition and resolution are measured separately, because an end-to-end
number cannot say which half moved. Resolution is measured with the gold spans
fed in through ``ManualRecognizer``, so the resolver is judged on the same
toponyms regardless of what the recognizer would have found.

    uv run python scripts/benchmark.py --limit 25
    uv run python scripts/benchmark.py --pipeline swapped --limit 25
"""

from __future__ import annotations

import argparse
import json
import os
import time
import typing as t
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_URL = (
    "https://raw.githubusercontent.com/milangritta/"
    "Pragmatic-Guide-to-Geoparsing-Evaluation/master/data/Corpora/GeoVirus.xml"
)
CORPUS_NAME = "GeoVirus"
# GeoVirus offsets sit one character past the text it ships; see parse_corpus.
OFFSET_SHIFT = -1
GAZETTEER_NAME = "geonames"

if t.TYPE_CHECKING:
    from geoparser.evaluation import Annotation


@dataclass(frozen=True)
class GoldSpan:
    """One gold toponym: where it is written, and where on Earth it is."""

    start: int
    end: int
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Document:
    """One corpus article and its gold toponyms."""

    identifier: str
    text: str
    gold: tuple[GoldSpan, ...]


@dataclass
class PipelineResult:
    """What one pipeline scored, and how long it took."""

    name: str
    models: dict[str, str]
    recognition: dict[str, float] = field(default_factory=dict)
    resolution: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def download_corpus(cache_path: Path) -> Path:
    """
    Return the corpus file, fetching it once into the cache.

    Args:
        cache_path: Where the downloaded XML is kept between runs

    Returns:
        The path to the corpus file on disk
    """
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CORPUS_URL, timeout=120) as response:
            cache_path.write_bytes(response.read())
    return cache_path


def parse_corpus(path: Path, *, limit: int | None = None) -> list[Document]:
    """
    Read GeoVirus into documents with gold spans.

    GeoVirus records its offsets one character further along than the text it
    ships: ``start`` 170 for "Pandi" indexes "andi,". The shift is uniform --
    every one of the 2167 gold spans in all 229 articles aligns at -1, and no
    other shift in [-3, 3] aligns any -- so it is corrected here rather than
    silently costing every pipeline its recall.

    A span that still does not match the text it points at is dropped, because
    it could only ever count as a miss and would depress every pipeline alike.

    Args:
        path: The corpus XML file
        limit: Keep only the first this many articles, for a shorter run

    Returns:
        The parsed documents, in corpus order
    """
    root = ET.parse(path).getroot()
    documents = []
    for index, article in enumerate(root.findall("article")):
        text = article.findtext("text") or ""
        if not text:
            continue
        gold = []
        for location in article.findall("./locations/location"):
            name = (location.findtext("name") or "").strip()
            start = int(location.findtext("start") or -1) + OFFSET_SHIFT
            end = int(location.findtext("end") or -1) + OFFSET_SHIFT
            latitude = location.findtext("lat")
            longitude = location.findtext("lon")
            if start < 0 or latitude is None or longitude is None:
                continue
            if text[start:end] != name:
                continue  # offsets do not match this text; not a gold span here
            gold.append(GoldSpan(start, end, name, float(latitude), float(longitude)))
        if gold:
            documents.append(Document(str(index), text, tuple(gold)))
        if limit is not None and len(documents) >= limit:
            break
    return documents


def gold_annotations(documents: Sequence[Document]) -> list[Annotation]:
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


def predicted_annotations(
    project: t.Any, document_ids: Sequence[t.Any], documents: Sequence[Document]
) -> list[Annotation]:
    """
    Read predictions back, tagged with the document they belong to.

    Documents are asked for by the IDs ``create_documents`` returned, because
    an unfiltered read has no ordering contract.
    """
    from geoparser.evaluation import Annotation

    annotations = []
    for document, parsed in zip(
        documents, project.get_documents(list(document_ids)), strict=True
    ):
        for toponym in parsed.toponyms:
            latitude, longitude = _coordinates(toponym.location)
            annotations.append(
                Annotation(
                    toponym.start,
                    toponym.end,
                    getattr(toponym.location, "identifier", None),
                    document.identifier,
                    latitude,
                    longitude,
                )
            )
    return annotations


def score_recognition(
    expected: Sequence[Annotation], predicted: Sequence[Annotation]
) -> dict[str, float]:
    """Return precision, recall and F1 over exactly matched spans."""
    from geoparser.evaluation import (
        recognition_f1,
        recognition_precision,
        recognition_recall,
    )

    return {
        "precision": recognition_precision(expected, predicted),
        "recall": recognition_recall(expected, predicted),
        "f1": recognition_f1(expected, predicted),
    }


def score_resolution(
    expected: Sequence[Annotation], predicted: Sequence[Annotation]
) -> dict[str, float]:
    """Return the distance metrics for a set of placed toponyms."""
    from geoparser.evaluation import (
        accuracy_at_km,
        area_under_error_curve,
        mean_error_km,
        median_error_km,
    )

    return {
        "accuracy_at_161km": accuracy_at_km(expected, predicted),
        "mean_error_km": mean_error_km(expected, predicted),
        "median_error_km": median_error_km(expected, predicted),
        "auc": area_under_error_curve(expected, predicted),
    }


def build_recognizer(pipeline: str) -> t.Any:
    """Return the recognizer half of a named pipeline."""
    if pipeline == "upstream":
        from geoparser.modules import SpacyRecognizer

        return SpacyRecognizer(model_name="en_core_web_sm")
    from geoparser.modules import GLiNER2Recognizer

    return GLiNER2Recognizer()


def build_resolver(pipeline: str) -> t.Any:
    """Return the resolver half of a named pipeline."""
    if pipeline == "upstream":
        from geoparser.modules import SentenceTransformerResolver

        return SentenceTransformerResolver(gazetteer_name=GAZETTEER_NAME)
    from geoparser.modules import JinaResolver

    return JinaResolver(gazetteer_name=GAZETTEER_NAME)


def run_pipeline(
    pipeline: str, documents: Sequence[Document], *, resolution_only: bool = False
) -> PipelineResult:
    """
    Run one pipeline over the corpus and score both halves.

    Recognition is scored from the pipeline's own spans. Resolution is scored
    from a second pass in which the gold spans are supplied directly, so the
    resolver is judged on the same toponyms in both pipelines.

    Args:
        pipeline: Either ``upstream`` or ``swapped``
        documents: The corpus to run
        resolution_only: Skip the recognition pass

    Returns:
        The scores and wall-clock time for this pipeline
    """
    import gc

    from geoparser.modules import ManualRecognizer
    from geoparser.project import Project

    expected = gold_annotations(documents)
    texts = [document.text for document in documents]
    result = PipelineResult(name=pipeline, models={})
    started = time.perf_counter()

    if not resolution_only:
        project = Project(f"bench-rec-{uuid.uuid4().hex[:8]}")
        try:
            document_ids = project.create_documents(texts)
            recognizer = build_recognizer(pipeline)
            result.models["recognizer"] = recognizer.model_name
            project.run_recognizer(recognizer)
            del recognizer
            gc.collect()
            predicted = predicted_annotations(project, document_ids, documents)
            result.recognition = score_recognition(expected, predicted)
        finally:
            project.delete()

    project = Project(f"bench-res-{uuid.uuid4().hex[:8]}")
    try:
        document_ids = project.create_documents(texts)
        references = [
            [(span.start, span.end) for span in document.gold] for document in documents
        ]
        project.run_recognizer(ManualRecognizer("gold", texts, references))
        resolver = build_resolver(pipeline)
        result.models["resolver"] = resolver.model_name
        reranker = getattr(resolver, "reranker_name", None)
        if reranker:
            result.models["reranker"] = reranker
        project.run_resolver(resolver)
        del resolver
        gc.collect()
        predicted = predicted_annotations(project, document_ids, documents)
        result.resolution = score_resolution(expected, predicted)
    finally:
        project.delete()

    result.elapsed_seconds = time.perf_counter() - started
    return result


def render_markdown(
    results: Sequence[PipelineResult], documents: Sequence[Document]
) -> str:
    """Render the comparison as a report."""
    gold_count = sum(len(document.gold) for document in documents)
    lines = [
        f"# {CORPUS_NAME} benchmark",
        "",
        f"- Documents: {len(documents)}",
        f"- Gold toponyms: {gold_count}",
        f"- Gazetteer: {GAZETTEER_NAME}",
        "",
        "## Recognition (pipeline's own spans, exact match)",
        "",
        "| Pipeline | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        if not result.recognition:
            continue
        lines.append(
            f"| {result.name} | {result.recognition['precision']:.3f} "
            f"| {result.recognition['recall']:.3f} "
            f"| {result.recognition['f1']:.3f} |"
        )
    lines += [
        "",
        "## Resolution (gold spans supplied, distance scored)",
        "",
        "| Pipeline | Acc@161km | Mean err (km) | Median err (km) | AUC | Seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        if not result.resolution:
            continue
        lines.append(
            f"| {result.name} | {result.resolution['accuracy_at_161km']:.3f} "
            f"| {result.resolution['mean_error_km']:.1f} "
            f"| {result.resolution['median_error_km']:.1f} "
            f"| {result.resolution['auc']:.3f} "
            f"| {result.elapsed_seconds:.1f} |"
        )
    lines += [
        "",
        "Acc@161km and the error figures charge a gold toponym the pipeline did",
        "not place the maximum possible error, so resolving less cannot improve",
        "a score. AUC is a log-scaled summary of the whole error distribution,",
        "lower being better; it is this harness's own normalization, so compare",
        "runs of this script against each other rather than against published",
        "AUC figures.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        action="append",
        choices=["upstream", "swapped"],
        help="Pipeline to run; repeat for several. Defaults to both.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Use only the first N articles"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--resolution-only", action="store_true")
    arguments = parser.parse_args(argv)

    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GEOPARSER_DB_PATH", str(output_dir / "benchmark.sqlite"))

    corpus = download_corpus(output_dir / f"{CORPUS_NAME}.xml")
    documents = parse_corpus(corpus, limit=arguments.limit)
    gold_count = sum(len(document.gold) for document in documents)
    print(f"{len(documents)} documents, {gold_count} gold toponyms")

    results = []
    for pipeline in arguments.pipeline or ["upstream", "swapped"]:
        print(f"running {pipeline}...", flush=True)
        result = run_pipeline(
            pipeline, documents, resolution_only=arguments.resolution_only
        )
        results.append(result)
        print(f"  {result.recognition} {result.resolution}", flush=True)

    report = render_markdown(results, documents)
    (output_dir / "benchmark-report.md").write_text(report)
    (output_dir / "benchmark-report.json").write_text(
        json.dumps(
            {
                "corpus": CORPUS_NAME,
                "documents": len(documents),
                "gold_toponyms": gold_count,
                "gazetteer": GAZETTEER_NAME,
                "pipelines": [
                    {
                        "name": result.name,
                        "models": result.models,
                        "recognition": result.recognition,
                        "resolution": result.resolution,
                        "elapsed_seconds": result.elapsed_seconds,
                    }
                    for result in results
                ],
            },
            indent=2,
        )
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
