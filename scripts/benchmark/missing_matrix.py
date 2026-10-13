"""Frozen first-five-corpora cells and checkpoint provenance for the comparison."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CorpusLanguageCell:
    """One fixed corpus and language slice with its verified size."""

    corpus: str
    language: str
    documents: int
    gold_toponyms: int


MATRIX = (
    CorpusLanguageCell("geovirus", "en", 229, 2167),
    CorpusLanguageCell("hipe2020", "de", 48, 558),
    CorpusLanguageCell("hipe2020", "en", 38, 158),
    CorpusLanguageCell("hipe2020", "fr", 43, 800),
    CorpusLanguageCell("newseye", "de", 8, 628),
    CorpusLanguageCell("newseye", "fi", 18, 214),
    CorpusLanguageCell("newseye", "fr", 33, 665),
    CorpusLanguageCell("newseye", "sv", 18, 265),
    CorpusLanguageCell("topres19th", "en", 110, 880),
    CorpusLanguageCell("newsli", "ar", 500, 3319),
    CorpusLanguageCell("newsli", "de", 500, 1019),
    CorpusLanguageCell("newsli", "es", 500, 1270),
    CorpusLanguageCell("newsli", "fa", 71, 326),
    CorpusLanguageCell("newsli", "ja", 500, 1577),
    CorpusLanguageCell("newsli", "pl", 186, 196),
    CorpusLanguageCell("newsli", "ro", 226, 385),
    CorpusLanguageCell("newsli", "sr", 500, 985),
    CorpusLanguageCell("newsli", "ta", 500, 1578),
    CorpusLanguageCell("newsli", "tr", 500, 1578),
    CorpusLanguageCell("newsli", "uk", 164, 258),
)

MISSING_PIPELINES = ("swapped", "xlm-minilm", "otter-cross-minilm")

# Immutable Hub revisions resolved from the public model API on 2026-10-13.
MODEL_PINS = {
    "gliner": (
        "fastino/gliner2.5-multi-v1",
        "cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3",
    ),
    "jina_embeddings": (
        "jinaai/jina-embeddings-v5-text-small",
        "dd76d535f5447ca3897a9c893fb1e612ead98192",
    ),
    "jina_reranker": (
        "jinaai/jina-reranker-v3.5",
        "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
    ),
    "minilm": (
        "dguzh/geo-all-MiniLM-L6-v2",
        "d678769f195c194ffdc6f3735a9360118a855b43",
    ),
    "xlm_r": (
        "Davlan/xlm-roberta-base-ner-hrl",
        "253f557bd8249b8515114cfd7f71974fe5fa4d2f",
    ),
    "otter_cross": (
        "whoisjones/otter-cross-mmbert",
        "8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d",
    ),
}

MODEL_METADATA = {
    "gliner": {
        "license": "Apache-2.0",
        "language_support": "multilingual claim; exhaustive list not published",
    },
    "jina_embeddings": {
        "license": "CC-BY-NC-4.0",
        "language_support": "119+ languages claimed; exhaustive list not published",
    },
    "jina_reranker": {
        "license": "CC-BY-NC-4.0",
        "language_support": "multilingual claim; exhaustive list not published",
    },
    "minilm": {
        "license": "not declared in public Hub metadata",
        "language_support": "model card does not provide an exhaustive list",
    },
    "xlm_r": {
        "license": "AFL-3.0",
        "language_support": (
            "fine-tuned for Arabic, German, English, Spanish, French, Italian, "
            "Latvian, Dutch, Portuguese, and Chinese; other languages are "
            "cross-lingual transfer"
        ),
    },
    "otter_cross": {
        "license": "Apache-2.0",
        "language_support": "multilingual claim; supported-language list unspecified",
    },
}

XLM_FINETUNED_LANGUAGES = frozenset(
    {"ar", "de", "en", "es", "fr", "it", "lv", "nl", "pt", "zh"}
)


def language_coverage(model: str, language: str) -> str:
    """Report documented coverage without treating unlisted transfer as unsupported."""
    if model == "xlm_r":
        return (
            "fine_tuned"
            if language in XLM_FINETUNED_LANGUAGES
            else "cross_lingual_transfer"
        )
    if model == "otter_cross":
        return "unspecified"
    if model in {"gliner", "jina_embeddings", "jina_reranker", "minilm"}:
        return "multilingual_claim"
    message = f"Unknown model key: {model}"
    raise KeyError(message)


def resolver_family(pipeline: str) -> str:
    """Return the resolver whose identity determines gold-span reuse."""
    if pipeline in {"upstream", "hybrid", "xlm-minilm", "otter-cross-minilm"}:
        return "minilm"
    if pipeline == "swapped":
        return "jina"
    if pipeline in {
        "prior",
        "trim",
        "population",
        "population-0.05",
        "population-0.2",
        "population-0.3",
        "population-0.5",
        "population-1.0",
    }:
        return "prior"
    message = f"Unknown benchmark pipeline: {pipeline}"
    raise KeyError(message)


def resolver_pins(pipeline: str) -> tuple[tuple[str, str], ...]:
    """Return exact Hub pins for this pipeline's resolver and optional reranker."""
    family = resolver_family(pipeline)
    if family == "minilm":
        return (MODEL_PINS["minilm"],)
    if family == "jina":
        return (MODEL_PINS["jina_embeddings"], MODEL_PINS["jina_reranker"])
    return (MODEL_PINS["minilm"],)


def pipeline_model_metadata(pipeline: str, language: str) -> dict[str, dict[str, str]]:
    """Return license and documented language evidence for each model role."""
    roles = {
        "swapped": {
            "recognizer": "gliner",
            "resolver": "jina_embeddings",
            "reranker": "jina_reranker",
        },
        "xlm-minilm": {"recognizer": "xlm_r", "resolver": "minilm"},
        "otter-cross-minilm": {
            "recognizer": "otter_cross",
            "resolver": "minilm",
        },
    }
    try:
        selected = roles[pipeline]
    except KeyError as error:
        message = f"No frozen missing-cell metadata for {pipeline!r}"
        raise KeyError(message) from error
    return {
        role: {
            "repository": MODEL_PINS[key][0],
            "revision": MODEL_PINS[key][1],
            **MODEL_METADATA[key],
            "language_coverage": language_coverage(key, language),
        }
        for role, key in selected.items()
    }


def cell_name(cell: CorpusLanguageCell) -> str:
    """Return the benchmark CLI name for a matrix cell."""
    return (
        cell.corpus if cell.corpus == "geovirus" else f"{cell.corpus}-{cell.language}"
    )
