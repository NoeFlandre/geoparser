"""
Build the benchmark pipelines and place them on a device.

``upstream`` is what the library shipped before the module swap: spaCy for
recognition and the thesis' fine-tuned MiniLM for resolution. ``swapped`` is
the GLiNER2 and Jina v5 pair. ``hybrid`` keeps GLiNER2 recognition but uses
the upstream MiniLM resolver. ``prior`` is ``hybrid`` with the
``PriorResolver``: the same MiniLM encoder, plus a population prior on the
ranking. Its default weight is 0.3 and inflection fallback is off. Seven
ablation pipelines vary these options and are available by name, but are not
included in the default run.

Device placement is done here rather than left to the libraries because they
do not agree. ``SentenceTransformer`` selects CUDA by itself when it is
available, but the Jina reranker is loaded through ``AutoModel`` and stays
wherever it was built -- which on a GPU node means the most expensive model in
the comparison would quietly run on the CPU.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import typing as t
from collections.abc import Sequence
from pathlib import Path

from scripts.benchmark import missing_matrix

UPSTREAM = "upstream"
SWAPPED = "swapped"
HYBRID = "hybrid"
PRIOR = "prior"
XLM_MINILM = "xlm-minilm"
OTTER_MINILM = "otter-cross-minilm"
DEFAULT_PIPELINES = (UPSTREAM, SWAPPED, HYBRID, PRIOR)

# PriorResolver settings as (population_weight, inflection_fallback). The
# default prior uses population weight 0.3 with fallback off. The ablations
# vary one factor at a time,
# plus a sweep of the prior's weight on its own. Their recognition is
# hybrid's, so only their resolution phase needs running.
PRIOR_SETTINGS = {
    PRIOR: (0.3, False),
    "trim": (0.0, True),
    "population": (0.1, False),
    "population-0.05": (0.05, False),
    "population-0.2": (0.2, False),
    "population-0.3": (0.3, False),
    "population-0.5": (0.5, False),
    "population-1.0": (1.0, False),
}
ABLATIONS = tuple(name for name in PRIOR_SETTINGS if name != PRIOR)
PIPELINES = (*DEFAULT_PIPELINES, XLM_MINILM, OTTER_MINILM, *ABLATIONS)
UPSTREAM_RESOLVER_MODEL = "dguzh/geo-all-MiniLM-L6-v2"

# The upstream recognizer. The transformer pipeline needs a plugin the library
# does not install, so the small pipeline is what upstream runs out of the box.
UPSTREAM_SPACY_MODEL = "en_core_web_sm"
SPACY_CONTEXT_MODEL = "xx_sent_ud_sm"
SPACY_CONTEXT_MODEL_VERSION = "3.8.0"
XLM_BATCH_SIZE = 8
XLM_STRIDE = 128
OTTER_ENTITY_TYPES = ("city", "country", "location")
OTTER_THRESHOLD = 0.5
OTTER_BATCH_SIZE = 8
OTTER_ARM_PLAN = (
    Path(__file__).resolve().parents[2] / "docs/examples/otter-recognition-arms.json"
)


def _snapshot(model_key: str) -> Path:
    """Fetch one pinned Hub snapshot into the configured node-local cache."""
    repo_id, revision = missing_matrix.MODEL_PINS[model_key]
    from huggingface_hub import snapshot_download

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    path = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(hf_home / "hub"),
        )
    )
    if path.name != revision:
        message = f"{repo_id} resolved to snapshot {path.name}, expected {revision}"
        raise RuntimeError(message)
    return path


@contextlib.contextmanager
def _offline_model_load() -> t.Iterator[None]:
    """Prevent pinned local snapshots from falling back to unpinned downloads."""
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _require_context_model() -> None:
    """Fail before inference if the resolver's locked spaCy model is absent."""
    installed = _installed_context_model_version()
    if installed != SPACY_CONTEXT_MODEL_VERSION:
        message = (
            f"Expected {SPACY_CONTEXT_MODEL} {SPACY_CONTEXT_MODEL_VERSION}; "
            f"found {installed}"
        )
        raise RuntimeError(message)


def _installed_context_model_version() -> str:
    """Read the required splitter package version with a useful error."""
    try:
        return importlib.metadata.version("xx-sent-ud-sm")
    except importlib.metadata.PackageNotFoundError as error:
        message = (
            f"Install the locked {SPACY_CONTEXT_MODEL}=={SPACY_CONTEXT_MODEL_VERSION} "
            "package before running the pinned resolver comparison."
        )
        raise RuntimeError(message) from error


def _pinned_name(model_key: str) -> str:
    """Return the immutable repository and revision recorded in reports."""
    repo_id, revision = missing_matrix.MODEL_PINS[model_key]
    return f"{repo_id}@{revision}"


_CONTEXT_MODEL_PIN = f"{SPACY_CONTEXT_MODEL}@{SPACY_CONTEXT_MODEL_VERSION}"
_PIPELINE_MODEL_PINS = {
    UPSTREAM: {
        "recognizer": f"{UPSTREAM_SPACY_MODEL}@3.8.0",
        "resolver": _pinned_name("minilm"),
        "context_sentence_splitter": _CONTEXT_MODEL_PIN,
    },
    SWAPPED: {
        "recognizer": _pinned_name("gliner"),
        "resolver": _pinned_name("jina_embeddings"),
        "reranker": _pinned_name("jina_reranker"),
        "context_sentence_splitter": _CONTEXT_MODEL_PIN,
    },
    XLM_MINILM: {
        "recognizer": _pinned_name("xlm_r"),
        "resolver": _pinned_name("minilm"),
        "context_sentence_splitter": _CONTEXT_MODEL_PIN,
    },
    OTTER_MINILM: {
        "recognizer": _pinned_name("otter_cross"),
        "resolver": _pinned_name("minilm"),
        "context_sentence_splitter": _CONTEXT_MODEL_PIN,
    },
}
_PIPELINE_MODEL_PINS.update(
    {
        pipeline: {
            "recognizer": _pinned_name("gliner"),
            "resolver": _pinned_name("minilm"),
            "context_sentence_splitter": _CONTEXT_MODEL_PIN,
        }
        for pipeline in (HYBRID, *PRIOR_SETTINGS)
    }
)


def _attach_model_pin(model: t.Any, model_key: str) -> None:
    """Annotate a loaded object without changing its runtime model path."""
    model.benchmark_model_pin = _pinned_name(model_key)


def _otter_plan_sources() -> tuple[dict, dict, dict[str, str]]:
    """Read the Otter plan and verify its reviewed source files."""
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(OTTER_ARM_PLAN.read_text(encoding="utf-8"))
    if plan.get("format") != "otter-recognition-arm-plan-v1":
        message = "Unexpected Otter arm plan format."
        raise RuntimeError(message)
    manifest_path = root / plan["source_manifest"]
    review_path = root / plan["source_review"]
    digests = {
        "arm_plan_sha256": hashlib.sha256(OTTER_ARM_PLAN.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "source_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
    }
    if digests["source_manifest_sha256"] != plan["source_manifest_sha256"]:
        message = "Otter source manifest does not match its reviewed hash."
        raise RuntimeError(message)
    if digests["source_review_sha256"] != plan["source_review_sha256"]:
        message = "Otter source review does not match its reviewed hash."
        raise RuntimeError(message)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return plan, manifest, digests


def _otter_arm(plan: dict) -> dict:
    """Find the single retained cross encoder and verify its runtime settings."""
    repo_id, revision = missing_matrix.MODEL_PINS["otter_cross"]
    arms = [arm for arm in plan["arms"] if arm.get("key") == "otter_cross_mmbert"]
    if len(arms) != 1:
        message = "Otter arm plan must contain exactly one cross-encoder arm."
        raise RuntimeError(message)
    arm = arms[0]
    expected = {
        "model": (repo_id, revision),
        "architecture": "cross_encoder",
        "max_seq_length": 1024,
        "max_span_length": 30,
        "threshold": OTTER_THRESHOLD,
        "batch_size": OTTER_BATCH_SIZE,
        "entity_types": list(OTTER_ENTITY_TYPES),
    }
    actual = {
        "model": (arm["model"]["identifier"], arm["model"]["revision"]),
        "architecture": arm["expected_architecture"],
        "max_seq_length": arm["expected_max_seq_length"],
        "max_span_length": arm["expected_max_span_length"],
        "threshold": arm["threshold"],
        "batch_size": arm["batch_size"],
        "entity_types": arm["entity_types"],
    }
    if actual != expected:
        message = f"Otter arm plan differs from the benchmark integration: {actual!r}"
        raise RuntimeError(message)

    return arm


def _otter_weight_digest(manifest: dict, arm: dict) -> str:
    """Match the selected Otter weights to their source manifest digest."""
    repo_id, revision = missing_matrix.MODEL_PINS["otter_cross"]
    entry = _otter_manifest_model(manifest, repo_id, revision)
    weight = _otter_manifest_weight(entry)
    if weight["sha256"] != arm["model"]["sha256"]:
        message = "Otter arm weight digest differs from the source manifest."
        raise RuntimeError(message)
    return weight["sha256"]


def _otter_manifest_model(manifest: dict, repo_id: str, revision: str) -> dict:
    """Select the exact model revision from a retained source manifest."""
    return next(
        model
        for model in manifest["models"]
        if (model["identifier"], model["revision"]) == (repo_id, revision)
    )


def _otter_manifest_weight(entry: dict) -> dict:
    """Find the safetensors weight file needed by the retained arm."""
    weight = next(
        (
            artifact
            for artifact in entry["files"]
            if artifact["path"] == "model.safetensors"
        ),
        None,
    )
    if weight is None:
        message = "Otter source manifest does not contain model.safetensors."
        raise RuntimeError(message)
    return weight


def _otter_arm_plan() -> tuple[dict, dict, dict[str, str]]:
    """Load and verify the retained Otter arm configuration and source review."""
    plan, manifest, digests = _otter_plan_sources()
    arm = _otter_arm(plan)
    weight_sha256 = _otter_weight_digest(manifest, arm)
    return plan, arm, {**digests, "model_sha256": weight_sha256}


def otter_integration_provenance() -> dict[str, t.Any]:
    """Return verified Otter integration hashes and stale-lock disclosure."""
    plan, arm, digests = _otter_arm_plan()
    root = Path(__file__).resolve().parents[2]
    current_lock = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    return {
        **digests,
        "model": arm["source"],
        "license": "Apache-2.0",
        "plan_status": plan["status"],
        "plan_dependency_lock_sha256": plan["dependency_lock_sha256"],
        "current_dependency_lock_sha256": current_lock,
        "plan_lock_matches_current": plan["dependency_lock_sha256"] == current_lock,
        "language_support": arm["language_support"],
        "training_overlap": arm["training_overlap"],
    }


def _otter_source_entry() -> tuple[dict, dict]:
    """Load the pinned Otter model manifest and its source import closure."""
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "docs/examples/otter-source-manifest.json").read_text(encoding="utf-8")
    )
    repo_id, revision = missing_matrix.MODEL_PINS["otter_cross"]
    entry = next(
        model
        for model in manifest["models"]
        if (model["identifier"], model["revision"]) == (repo_id, revision)
    )
    return manifest, entry


def _verify_otter_artifact(snapshot: Path, artifact: dict) -> str:
    """Check one downloaded artifact's byte count and SHA-256."""
    path = snapshot / artifact["path"]
    if not path.is_file() or path.stat().st_size != artifact["bytes"]:
        message = f"Otter artifact missing or wrong size: {artifact['path']}"
        raise RuntimeError(message)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != artifact["sha256"]:
        message = f"Otter artifact hash mismatch: {artifact['path']}"
        raise RuntimeError(message)
    return artifact["path"]


def _verify_otter_import_closure(manifest: dict, verified: set[str]) -> None:
    """Require every dynamically imported project file to be hash-verified."""
    missing_code = set(manifest["inference_imports"]) - verified
    if missing_code:
        message = f"Otter import closure is not fully pinned: {sorted(missing_code)}"
        raise RuntimeError(message)


def _verify_otter_snapshot(snapshot: Path) -> None:
    """Verify the retained Otter plan, source closure and weights before loading code."""
    _otter_arm_plan()
    manifest, entry = _otter_source_entry()
    verified = {
        _verify_otter_artifact(snapshot, artifact) for artifact in entry["files"]
    }
    _verify_otter_import_closure(manifest, verified)


def _xlm_entity_spans(entities: Sequence[dict[str, t.Any]]) -> list[tuple[int, int]]:
    """Keep LOC entity groups and return unique character spans in order."""
    spans = set()
    for entity in entities:
        label = entity.get("entity_group", entity.get("entity", ""))
        if label.removeprefix("B-").removeprefix("I-") != "LOC":
            continue
        start, end = entity.get("start"), entity.get("end")
        if start is not None and end is not None:
            spans.add((int(start), int(end)))
    return sorted(spans)


def _predict_xlm_spans(
    inference_pipeline: t.Any, texts: list[str]
) -> list[list[tuple[int, int]] | None]:
    """Run pinned XLM-R and convert one entity list per input document."""
    predictions = inference_pipeline(texts, batch_size=XLM_BATCH_SIZE)
    if len(predictions) != len(texts):
        message = "XLM-R returned a different number of document predictions."
        raise RuntimeError(message)
    return [
        t.cast(list[tuple[int, int]] | None, _xlm_entity_spans(entities))
        for entities in predictions
    ]


def _verify_otter_model_config(model: t.Any) -> None:
    """Reject a checkpoint whose runtime configuration differs from the arm plan."""
    _, arm, _ = _otter_arm_plan()
    config = model.config
    expected = {
        "architecture": arm["expected_architecture"],
        "max_seq_length": arm["expected_max_seq_length"],
        "max_span_length": arm["expected_max_span_length"],
    }
    actual = {key: getattr(config, key, None) for key in expected}
    if actual != expected:
        message = (
            f"Otter checkpoint config differs from its retained arm plan: {actual!r}"
        )
        raise RuntimeError(message)


def _build_gliner_recognizer(device: str) -> t.Any:
    """Load GLiNER2 from the explicit snapshot used for the Jina comparison."""
    from geoparser.modules import GLiNER2Recognizer

    snapshot = _snapshot("gliner")
    with _offline_model_load():
        recognizer = GLiNER2Recognizer(model_name=str(snapshot))
    recognizer.model_name = _pinned_name("gliner")
    _attach_model_pin(recognizer, "gliner")
    _move(getattr(recognizer, "model", None), device)
    return recognizer


def _build_xlm_recognizer(device: str) -> t.Any:
    """Load the pinned multilingual XLM-R location tagger with overlap windows."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    from geoparser.modules.recognizers import Recognizer

    snapshot = _snapshot("xlm_r")
    model_id = _pinned_name("xlm_r")
    with _offline_model_load():
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), use_fast=True, local_files_only=True
        )
        model = AutoModelForTokenClassification.from_pretrained(
            str(snapshot), local_files_only=True
        )
        model.eval()
        inference_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            stride=XLM_STRIDE,
            device=0 if device.startswith("cuda") else -1,
        )

    class XLMBenchmarkRecognizer(Recognizer):
        """Map pinned XLM-R LOC groups to the project span representation."""

        NAME = "XLMBenchmarkRecognizer"

        def __init__(self):
            super().__init__(model_name=model_id)
            self.model_name = model_id
            self.inference_pipeline = inference_pipeline
            self.benchmark_model_pin = model_id

        def predict(self, texts: list[str]) -> list[list[tuple[int, int]] | None]:
            return _predict_xlm_spans(self.inference_pipeline, texts)

    return XLMBenchmarkRecognizer()


def _build_otter_recognizer(device: str) -> t.Any:
    """Load the hash-reviewed Otter cross encoder and its local tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    from geoparser.modules import OtterRecognizer

    snapshot = _snapshot("otter_cross")
    _verify_otter_snapshot(snapshot)
    with _offline_model_load():
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), trust_remote_code=False, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            str(snapshot),
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
        )
    _verify_otter_model_config(model)
    model._tokenizer = tokenizer
    model.eval()
    model.to(device)
    recognizer = OtterRecognizer(
        model,
        source=_pinned_name("otter_cross"),
        entity_types=OTTER_ENTITY_TYPES,
        threshold=OTTER_THRESHOLD,
        batch_size=OTTER_BATCH_SIZE,
    )
    recognizer.model_name = _pinned_name("otter_cross")
    _attach_model_pin(recognizer, "otter_cross")
    recognizer.invalid_predictions = []
    recognizer.unsupported_inputs = []
    return recognizer


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
        msg = (
            "CUDA was requested but torch reports no CUDA device. Check that "
            "the job reserved a GPU and that the CUDA build of torch is "
            "installed."
        )
        raise RuntimeError(msg)
    return requested


def describe_device(device: str) -> str:
    """Return a human-readable name for the device in use."""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return f"{device} ({torch.cuda.get_device_name(0)})"
    return device


def _build_spacy_recognizer(device: str) -> t.Any:
    """Build the upstream small spaCy recognizer, optionally enabling GPU."""
    from geoparser.modules import SpacyRecognizer

    if device.startswith("cuda"):
        import spacy

        with contextlib.suppress(Exception):
            spacy.require_gpu()
    return SpacyRecognizer(model_name=UPSTREAM_SPACY_MODEL)


_RECOGNIZER_BUILDERS = {
    UPSTREAM: _build_spacy_recognizer,
    SWAPPED: _build_gliner_recognizer,
    HYBRID: _build_gliner_recognizer,
    PRIOR: _build_gliner_recognizer,
    XLM_MINILM: _build_xlm_recognizer,
    OTTER_MINILM: _build_otter_recognizer,
    **dict.fromkeys(ABLATIONS, _build_gliner_recognizer),
}


def build_recognizer(pipeline: str, *, device: str) -> t.Any:
    """
    Return the recognizer half of a named pipeline.

    Args:
        pipeline: A named recognizer/resolver comparison
        device: Where to place the model

    Returns:
        The constructed recognizer
    """
    try:
        builder = _RECOGNIZER_BUILDERS[pipeline]
    except KeyError as error:
        msg = f"Unknown benchmark pipeline: {pipeline}"
        raise ValueError(msg) from error
    return builder(device)


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
        pipeline: ``upstream`` and ``hybrid`` use MiniLM, ``prior`` and
            the ablations MiniLM with PRIOR_SETTINGS, and ``swapped`` Jina
        device: Where to place the models
        min_similarity: Similarity the best candidate must reach to be used

    Returns:
        The constructed resolver
    """
    from geoparser.gazetteer import Gazetteer  # noqa: F401 - import order

    if pipeline not in PIPELINES:
        msg = f"Unknown benchmark pipeline: {pipeline}"
        raise ValueError(msg)

    if pipeline in (UPSTREAM, HYBRID):
        from geoparser.modules import SentenceTransformerResolver

        snapshot = _snapshot("minilm")
        with _offline_model_load():
            resolver = SentenceTransformerResolver(
                model_name=str(snapshot),
                gazetteer_name=GAZETTEER_NAME,
                min_similarity=min_similarity,
            )
        resolver.benchmark_model_pin = _pinned_name("minilm")
    elif pipeline in PRIOR_SETTINGS:
        from geoparser.modules import PriorResolver

        population_weight, inflection_fallback = PRIOR_SETTINGS[pipeline]
        snapshot = _snapshot("minilm")
        with _offline_model_load():
            resolver = PriorResolver(
                model_name=str(snapshot),
                gazetteer_name=GAZETTEER_NAME,
                min_similarity=min_similarity,
                population_weight=population_weight,
                inflection_fallback=inflection_fallback,
            )
        resolver.benchmark_model_pin = _pinned_name("minilm")
    elif pipeline == SWAPPED:
        from geoparser.modules import JinaResolver

        _require_context_model()
        embeddings_snapshot = _snapshot("jina_embeddings")
        reranker_snapshot = _snapshot("jina_reranker")
        with _offline_model_load():
            resolver = JinaResolver(
                model_name=str(embeddings_snapshot),
                reranker_name=str(reranker_snapshot),
                gazetteer_name=GAZETTEER_NAME,
                min_similarity=min_similarity,
            )
        resolver.benchmark_model_pin = _pinned_name("jina_embeddings")
        resolver.benchmark_reranker_pin = _pinned_name("jina_reranker")
    else:
        from geoparser.modules import SentenceTransformerResolver

        _require_context_model()
        snapshot = _snapshot("minilm")
        with _offline_model_load():
            resolver = SentenceTransformerResolver(
                model_name=str(snapshot),
                gazetteer_name=GAZETTEER_NAME,
                min_similarity=min_similarity,
            )
        resolver.benchmark_model_pin = _pinned_name("minilm")

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
        names["recognizer"] = getattr(
            recognizer, "benchmark_model_pin", recognizer.model_name
        )
    if resolver is not None:
        names["resolver"] = getattr(
            resolver, "benchmark_model_pin", resolver.model_name
        )
        reranker = getattr(
            resolver,
            "benchmark_reranker_pin",
            getattr(resolver, "reranker_name", None),
        )
        if reranker:
            names["reranker"] = reranker
    return names


def pipeline_model_pins(pipeline: str) -> dict[str, str]:
    """Return the immutable model identity that makes a checkpoint resumable."""
    try:
        return _PIPELINE_MODEL_PINS[pipeline].copy()
    except KeyError as error:
        message = f"Unknown benchmark pipeline: {pipeline}"
        raise ValueError(message) from error
