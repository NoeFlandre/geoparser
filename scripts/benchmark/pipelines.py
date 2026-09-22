"""
Build the benchmark pipelines and place them on a device.

``upstream`` is what the library shipped before the module swap: spaCy for
recognition and the thesis' fine-tuned MiniLM for resolution. ``swapped`` is
the GLiNER2 and Jina v5 pair. ``hybrid`` keeps GLiNER2 recognition but uses
the upstream MiniLM resolver.

Device placement is done here rather than left to the libraries because they
do not agree. ``SentenceTransformer`` selects CUDA by itself when it is
available, but the Jina reranker is loaded through ``AutoModel`` and stays
wherever it was built -- which on a GPU node means the most expensive model in
the comparison would quietly run on the CPU.
"""

from __future__ import annotations

import contextlib
import typing as t

UPSTREAM = "upstream"
SWAPPED = "swapped"
HYBRID = "hybrid"
PIPELINES = (UPSTREAM, SWAPPED, HYBRID)
UPSTREAM_RESOLVER_MODEL = "dguzh/geo-all-MiniLM-L6-v2"

# The upstream recognizer. The transformer pipeline needs a plugin the library
# does not install, so the small pipeline is what upstream runs out of the box.
UPSTREAM_SPACY_MODEL = "en_core_web_sm"


def resolve_device(requested: str) -> str:
    """
    Return the device to run on, checking that a requested GPU exists.

    Asking for CUDA and silently getting CPU is the failure this prevents: the
    run would still finish, many hours later, and nothing in its output would
    say why.

    Args:
        requested: ``cpu``, ``cuda``, or ``auto`` to prefer CUDA when present

    Returns:
        The device string to place models on

    Raises:
        RuntimeError: If CUDA was asked for explicitly and is not available
    """
    import torch

    available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested == "cuda" and not available:
        raise RuntimeError(
            "CUDA was requested but torch reports no CUDA device. Check that "
            "the job reserved a GPU and that the CUDA build of torch is "
            "installed."
        )
    return requested


def describe_device(device: str) -> str:
    """Return a human-readable name for the device in use."""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return f"{device} ({torch.cuda.get_device_name(0)})"
    return device


def build_recognizer(pipeline: str, *, device: str) -> t.Any:
    """
    Return the recognizer half of a named pipeline.

    Args:
        pipeline: ``upstream`` uses spaCy; ``swapped`` and ``hybrid`` use GLiNER2
        device: Where to place the model

    Returns:
        The constructed recognizer
    """
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown benchmark pipeline: {pipeline}")

    if pipeline == UPSTREAM:
        from geoparser.modules import SpacyRecognizer

        if device.startswith("cuda"):
            import spacy

            # Best effort: spaCy only uses the GPU when cupy is installed for
            # the right CUDA build, and the small pipeline barely benefits, so
            # its absence is not a reason to fail the run.
            with contextlib.suppress(Exception):
                spacy.require_gpu()
        return SpacyRecognizer(model_name=UPSTREAM_SPACY_MODEL)

    from geoparser.modules import GLiNER2Recognizer

    recognizer = GLiNER2Recognizer()
    _move(getattr(recognizer, "model", None), device)
    return recognizer


def build_resolver(pipeline: str, *, device: str, min_similarity: float) -> t.Any:
    """
    Return the resolver half of a named pipeline, on the given device.

    ``min_similarity`` is passed explicitly rather than left at its default,
    because the default is not comparable across models. A resolver abstains
    when its best candidate scores below the threshold, and the inherited
    default of 0.6 is calibrated for the upstream MiniLM model: measured on
    this corpus, Jina v5's best candidate scores 0.28-0.44, so at 0.6 it
    abstains on every toponym while having retrieved the right candidates all
    along.

    Args:
        pipeline: ``upstream`` and ``hybrid`` use MiniLM; ``swapped`` uses Jina
        device: Where to place the models
        min_similarity: Similarity the best candidate must reach to be used

    Returns:
        The constructed resolver
    """
    from geoparser.gazetteer import Gazetteer  # noqa: F401 - import order

    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown benchmark pipeline: {pipeline}")

    if pipeline in (UPSTREAM, HYBRID):
        from geoparser.modules import SentenceTransformerResolver

        resolver = SentenceTransformerResolver(
            model_name=UPSTREAM_RESOLVER_MODEL,
            gazetteer_name=GAZETTEER_NAME,
            min_similarity=min_similarity,
        )
    else:
        from geoparser.modules import JinaResolver

        resolver = JinaResolver(
            gazetteer_name=GAZETTEER_NAME, min_similarity=min_similarity
        )

    _move(getattr(resolver, "transformer", None), device)
    _move(getattr(resolver, "reranker", None), device)
    return resolver


GAZETTEER_NAME = "geonames"


def _move(model: t.Any, device: str) -> None:
    """
    Move a model to a device when it is one that can be moved.

    Args:
        model: A torch module, a SentenceTransformer, or None
        device: Where to place it
    """
    if model is None:
        return
    mover = getattr(model, "to", None)
    if callable(mover):
        mover(device)


def model_names(recognizer: t.Any, resolver: t.Any) -> dict[str, str]:
    """
    Return the checkpoints a pipeline actually loaded.

    Args:
        recognizer: The recognizer, or None when only resolution ran
        resolver: The resolver, or None when only recognition ran

    Returns:
        Role to checkpoint name, for the report and the checkpoint identity
    """
    names = {}
    if recognizer is not None:
        names["recognizer"] = recognizer.model_name
    if resolver is not None:
        names["resolver"] = resolver.model_name
        reranker = getattr(resolver, "reranker_name", None)
        if reranker:
            names["reranker"] = reranker
    return names
