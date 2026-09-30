"""Pinned CPU inference adapters for the three PAN-X recognizers."""

from __future__ import annotations

import importlib.metadata
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from scripts.panx_benchmark.constants import (
    BATCH_SIZE,
    GLINER_ENTITY_LABELS,
    GLINER_THRESHOLD,
    LOCATION_LABEL,
    SPACY_LOCATION_LABELS,
    ModelSpec,
)
from scripts.panx_benchmark.data import Span


class BatchPredictor(Protocol):
    """One model's location spans for an ordered batch of texts."""

    def predict_batch(self, texts: list[str]) -> list[set[Span]]:
        """Return character spans aligned with every input text."""


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Loaded predictor and its separately measured cold acquisition/load."""

    predictor: BatchPredictor
    download_seconds: float
    load_seconds: float
    cache_hit: bool | None
    checkpoint_path: str


class GLiNERPredictor:
    """Ask GLiNER2's project-default place labels and union them as LOC."""

    def __init__(self, model: Any):
        self.model = model

    def predict_batch(self, texts: list[str]) -> list[set[Span]]:
        """Run pinned GLiNER2 entity extraction for a batch."""
        results = self.model.batch_extract_entities(
            texts,
            list(GLINER_ENTITY_LABELS),
            batch_size=BATCH_SIZE,
            threshold=GLINER_THRESHOLD,
            include_spans=True,
        )
        return [_gliner_location_spans(result) for result in results]


class XLMRecognizer:
    """Group XLM-R token tags into character-offset location spans."""

    def __init__(self, inference_pipeline: Any):
        self.inference_pipeline = inference_pipeline

    def predict_batch(self, texts: list[str]) -> list[set[Span]]:
        """Run the Hugging Face token-classification pipeline on one batch."""
        results = self.inference_pipeline(texts, batch_size=BATCH_SIZE)
        return [_xlm_location_spans(result) for result in results]


class SpacyRecognizer:
    """Run the upstream English spaCy pipeline with its location labels."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def predict_batch(self, texts: list[str]) -> list[set[Span]]:
        """Run spaCy's batched document pipeline and keep its place entities."""
        return [
            {
                (entity.start_char, entity.end_char)
                for entity in document.ents
                if entity.label_ in SPACY_LOCATION_LABELS
            }
            for document in self.pipeline.pipe(texts, batch_size=BATCH_SIZE)
        ]


def _gliner_location_spans(result: dict[str, Any]) -> set[Span]:
    """Map GLiNER2 city, country, and location outputs to the gold LOC class."""
    entities = result.get("entities", {})
    spans = set()
    for label, matches in entities.items():
        if label.casefold() not in GLINER_ENTITY_LABELS:
            continue
        spans.update(
            (int(match["start"]), int(match["end"]))
            for match in matches
            if "start" in match and "end" in match
        )
    return spans


def _xlm_location_spans(result: list[dict[str, Any]]) -> set[Span]:
    """Keep aggregated XLM-R `LOC` predictions at original text offsets."""
    spans = set()
    for entity in result:
        label = entity.get("entity_group", entity.get("entity", ""))
        if label.removeprefix("B-").removeprefix("I-") != LOCATION_LABEL:
            continue
        start, end = entity.get("start"), entity.get("end")
        if start is not None and end is not None:
            spans.add((int(start), int(end)))
    return spans


def _snapshot(spec: ModelSpec, cache_dir: Path) -> tuple[str, float, bool]:
    """Download one exact Hub revision before separately timing model load."""
    from huggingface_hub import snapshot_download

    if spec.revision is None:
        message = f"{spec.key} is missing a pinned model revision"
        raise ValueError(message)
    model_cache = cache_dir / "hub"
    snapshot = (
        model_cache
        / f"models--{spec.model_id.replace('/', '--')}"
        / "snapshots"
        / spec.revision
    )
    cache_hit = snapshot.is_dir()
    started = time.perf_counter()
    local_path = snapshot_download(
        repo_id=spec.model_id,
        revision=spec.revision,
        cache_dir=str(model_cache),
    )
    return local_path, time.perf_counter() - started, cache_hit


def _load_gliner(spec: ModelSpec, cache_dir: Path) -> LoadedModel:
    """Download and load the pinned GLiNER2.5 multi checkpoint."""
    from gliner2 import AutoExtractor

    local_path, download_seconds, cache_hit = _snapshot(spec, cache_dir)
    started = time.perf_counter()
    model = AutoExtractor.from_pretrained(local_path, local_files_only=True)
    return LoadedModel(
        GLiNERPredictor(model),
        download_seconds,
        time.perf_counter() - started,
        cache_hit,
        local_path,
    )


def _load_xlmr(spec: ModelSpec, cache_dir: Path) -> LoadedModel:
    """Download and load the pinned XLM-R token-classification checkpoint."""
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    local_path, download_seconds, cache_hit = _snapshot(spec, cache_dir)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(local_path)
    model.eval()
    inference_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,
    )
    return LoadedModel(
        XLMRecognizer(inference_pipeline),
        download_seconds,
        time.perf_counter() - started,
        cache_hit,
        local_path,
    )


def _load_spacy(spec: ModelSpec) -> LoadedModel:
    """Load the installed upstream English model and check its version pin."""
    import spacy

    installed = importlib.metadata.version("en-core-web-sm")
    if installed != spec.revision:
        message = f"Expected en_core_web_sm {spec.revision}; found {installed}"
        raise RuntimeError(message)
    started = time.perf_counter()
    pipeline = spacy.load(spec.model_id)
    for component in ("tagger", "parser", "attribute_ruler", "lemmatizer"):
        if component in pipeline.pipe_names:
            pipeline.remove_pipe(component)
    return LoadedModel(
        SpacyRecognizer(pipeline),
        0.0,
        time.perf_counter() - started,
        None,
        f"installed distribution en-core-web-sm=={installed}",
    )


def load_model(spec: ModelSpec, cache_dir: Path) -> LoadedModel:
    """Load the correct pinned model adapter without exposing a device choice."""
    if spec.key == "spacy_en":
        return _load_spacy(spec)
    if spec.key == "gliner2_multi":
        return _load_gliner(spec, cache_dir)
    if spec.key == "xlmr_ner_hrl":
        return _load_xlmr(spec, cache_dir)
    message = f"Unknown PAN-X recognizer: {spec.key}"
    raise ValueError(message)
