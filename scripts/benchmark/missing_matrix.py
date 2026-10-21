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

CURRENT_PIPELINES = (
    "gliner-minilm",
    "gliner-jina",
    "xlm-minilm",
    "xlm-jina",
    "otter-minilm",
    "otter-jina",
)

# Every current row is one recognizer/resolver combination. These component
# keys are the keys in MODEL_PINS and are also used to decide safe phase reuse.
PIPELINE_COMPONENTS = {
    "gliner-minilm": {"recognizer": "gliner", "resolver": "minilm"},
    "gliner-jina": {"recognizer": "gliner", "resolver": "jina"},
    "xlm-minilm": {"recognizer": "xlm_r", "resolver": "minilm"},
    "xlm-jina": {"recognizer": "xlm_r", "resolver": "jina"},
    "otter-minilm": {"recognizer": "otter_cross", "resolver": "minilm"},
    "otter-jina": {"recognizer": "otter_cross", "resolver": "jina"},
}

# Historical IDs remain valid evidence keys. Values name the exact components
# and settings that those IDs represented; reports are never rewritten in place.
LEGACY_PIPELINE_ID_MAP = {
    "upstream": "spacy:en_core_web_sm + dguzh/geo-all-MiniLM-L6-v2",
    "swapped": (
        "fastino/gliner2.5-multi-v1 + jinaai/jina-embeddings-v5-text-small + "
        "jinaai/jina-reranker-v3.5"
    ),
    "hybrid": "fastino/gliner2.5-multi-v1 + dguzh/geo-all-MiniLM-L6-v2",
    "prior": (
        "fastino/gliner2.5-multi-v1 + dguzh/geo-all-MiniLM-L6-v2; "
        "population prior weight=0.3, inflection fallback=false"
    ),
    "trim": (
        "fastino/gliner2.5-multi-v1 + dguzh/geo-all-MiniLM-L6-v2; "
        "population prior weight=0, inflection fallback=true"
    ),
    "population": (
        "fastino/gliner2.5-multi-v1 + dguzh/geo-all-MiniLM-L6-v2; "
        "population prior weight=0.1, inflection fallback=false"
    ),
    "population-0.05": (
        "fastino/gliner2.5-multi-v1 + dguzh/geo-all-MiniLM-L6-v2; "
        "population prior weight=0.05, inflection fallback=false"
    ),
    "population-0.2": (
        "fastino/gliner2.5-multi-v1 + dguzh/geo-all-MiniLM-L6-v2; "
        "population prior weight=0.2, inflection fallback=false"
    ),
    "population-0.3": (
        "fastino/gliner2.5-multi-v1 + dguzh/geo-all-MiniLM-L6-v2; "
        "population prior weight=0.3, inflection fallback=false"
    ),
    "population-0.5": (
        "fastino/gliner2.5-multi-v1 + dguzh/geo-all-MiniLM-L6-v2; "
        "population prior weight=0.5, inflection fallback=false"
    ),
    "population-1.0": (
        "fastino/gliner2.5-multi-v1 + dguzh/geo-all-MiniLM-L6-v2; "
        "population prior weight=1.0, inflection fallback=false"
    ),
    "xlm-minilm": "Davlan/xlm-roberta-base-ner-hrl + dguzh/geo-all-MiniLM-L6-v2",
    "otter-cross-minilm": (
        "whoisjones/otter-cross-mmbert + dguzh/geo-all-MiniLM-L6-v2"
    ),
}

# Backward-compatible name for callers that used the earlier missing-cell
# subset. New runs use CURRENT_PIPELINES.
MISSING_PIPELINES = ("swapped", "xlm-minilm", "otter-cross-minilm")

NEWSLI_EXACT_DEDUP_LANGUAGES = ("ar", "de", "es", "fa", "ja", "ta", "tr")
NEWSLI_EXACT_DUPLICATES_REMOVED = {
    "ar": 471,
    "de": 4,
    "es": 18,
    "fa": 147,
    "ja": 49,
    "ta": 561,
    "tr": 216,
}

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
    if pipeline in PIPELINE_COMPONENTS:
        return PIPELINE_COMPONENTS[pipeline]["resolver"]
    legacy_component = {
        "upstream": "minilm",
        "swapped": "jina",
        "hybrid": "minilm",
        "otter-cross-minilm": "minilm",
    }
    if pipeline in legacy_component:
        return legacy_component[pipeline]
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


def recognizer_family(pipeline: str) -> str:
    """Return the model key that determines reusable recognition output."""
    if pipeline in PIPELINE_COMPONENTS:
        return PIPELINE_COMPONENTS[pipeline]["recognizer"]
    legacy = {
        "upstream": "spacy",
        "swapped": "gliner",
        "hybrid": "gliner",
        "prior": "gliner",
        "trim": "gliner",
        "population": "gliner",
        "population-0.05": "gliner",
        "population-0.2": "gliner",
        "population-0.3": "gliner",
        "population-0.5": "gliner",
        "population-1.0": "gliner",
        "otter-cross-minilm": "otter_cross",
    }
    if pipeline in legacy:
        return legacy[pipeline]
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
        pipeline: {
            "recognizer": components["recognizer"],
            **(
                {
                    "resolver": "jina_embeddings",
                    "reranker": "jina_reranker",
                }
                if components["resolver"] == "jina"
                else {"resolver": "minilm"}
            ),
        }
        for pipeline, components in PIPELINE_COMPONENTS.items()
    }
    roles.update(
        {
            "swapped": {
                "recognizer": "gliner",
                "resolver": "jina_embeddings",
                "reranker": "jina_reranker",
            },
            "hybrid": {"recognizer": "gliner", "resolver": "minilm"},
            "xlm-minilm": {"recognizer": "xlm_r", "resolver": "minilm"},
            "otter-cross-minilm": {
                "recognizer": "otter_cross",
                "resolver": "minilm",
            },
        }
    )
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
