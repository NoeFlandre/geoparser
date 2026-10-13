"""Factory tests for benchmark pipeline composition."""

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from scripts.benchmark import pipelines
from scripts.benchmark.__main__ import build_parser


def _patch_module_classes(monkeypatch, **classes):
    """Patch lazy module imports without loading transformer models."""
    package = ModuleType("geoparser")
    package.__path__ = []
    modules = ModuleType("geoparser.modules")

    def get_module_class(name: str) -> Any:
        if name not in classes:
            raise AttributeError(name)
        return classes[name]

    modules.__dict__["__getattr__"] = get_module_class
    gazetteer = ModuleType("geoparser.gazetteer")
    gazetteer.__dict__["Gazetteer"] = Mock()
    package.__dict__["modules"] = modules
    package.__dict__["gazetteer"] = gazetteer
    monkeypatch.setitem(sys.modules, "geoparser", package)
    monkeypatch.setitem(sys.modules, "geoparser.modules", modules)
    monkeypatch.setitem(sys.modules, "geoparser.gazetteer", gazetteer)


def test_hybrid_is_a_distinct_benchmark_pipeline():
    assert pipelines.DEFAULT_PIPELINES == ("upstream", "swapped", "hybrid", "prior")


def test_cli_accepts_the_hybrid_pipeline():
    arguments = build_parser().parse_args(["--pipeline", pipelines.HYBRID])

    assert arguments.pipeline == [pipelines.HYBRID]


@pytest.mark.parametrize("pipeline", [pipelines.XLM_MINILM, pipelines.OTTER_MINILM])
def test_cli_accepts_each_missing_recognizer_pipeline(pipeline):
    arguments = build_parser().parse_args(["--pipeline", pipeline])

    assert arguments.pipeline == [pipeline]


def test_hybrid_uses_gliner2_for_recognition(monkeypatch, tmp_path):
    recognizer = SimpleNamespace(model_name="fastino/gliner2.5-multi-v1", model=Mock())
    gliner_factory = Mock(return_value=recognizer)
    snapshot = tmp_path / "gliner-snapshot"
    _patch_module_classes(monkeypatch, GLiNER2Recognizer=gliner_factory)
    monkeypatch.setattr(pipelines, "_snapshot", lambda key: snapshot)

    result = pipelines.build_recognizer(pipelines.HYBRID, device="cuda")

    assert result is recognizer
    gliner_factory.assert_called_once_with(model_name=str(snapshot))
    assert recognizer.benchmark_model_pin == pipelines._pinned_name("gliner")
    recognizer.model.to.assert_called_once_with("cuda")


@pytest.mark.parametrize("pipeline", [pipelines.UPSTREAM, "hybrid"])
def test_upstream_and_hybrid_use_the_same_minilm_resolver(monkeypatch, pipeline):
    snapshot = Path("/cache/minilm-snapshot")
    transformer = Mock()
    resolver = SimpleNamespace(
        model_name="dguzh/geo-all-MiniLM-L6-v2",
        transformer=transformer,
        reranker=None,
    )
    resolver_factory = Mock(return_value=resolver)
    _patch_module_classes(monkeypatch, SentenceTransformerResolver=resolver_factory)
    monkeypatch.setattr(pipelines, "GAZETTEER_NAME", "geonames")
    snapshot_factory = Mock(return_value=snapshot)
    monkeypatch.setattr(pipelines, "_snapshot", snapshot_factory)

    result = pipelines.build_resolver(pipeline, device="cuda", min_similarity=0.0)

    assert result is resolver
    snapshot_factory.assert_called_once_with("minilm")
    resolver_factory.assert_called_once_with(
        model_name=str(snapshot),
        gazetteer_name="geonames",
        min_similarity=0.0,
    )
    assert resolver.benchmark_model_pin == pipelines._pinned_name("minilm")
    transformer.to.assert_called_once_with("cuda")


def test_swapped_keeps_jina_resolver(monkeypatch):
    resolver = SimpleNamespace(transformer=Mock(), reranker=Mock())
    jina_factory = Mock(return_value=resolver)
    _patch_module_classes(monkeypatch, JinaResolver=jina_factory)
    snapshots = {
        "jina_embeddings": Path("/cache/jina-embeddings"),
        "jina_reranker": Path("/cache/jina-reranker"),
    }
    monkeypatch.setattr(pipelines, "_require_context_model", Mock())
    monkeypatch.setattr(pipelines, "_snapshot", lambda key: snapshots[key])

    result = pipelines.build_resolver(
        pipelines.SWAPPED, device="cpu", min_similarity=0.0
    )

    assert result is resolver
    jina_factory.assert_called_once_with(
        model_name=str(snapshots["jina_embeddings"]),
        reranker_name=str(snapshots["jina_reranker"]),
        gazetteer_name="geonames",
        min_similarity=0.0,
    )
    assert resolver.benchmark_model_pin == pipelines._pinned_name("jina_embeddings")
    assert resolver.benchmark_reranker_pin == pipelines._pinned_name("jina_reranker")
    resolver.transformer.to.assert_called_once_with("cpu")
    resolver.reranker.to.assert_called_once_with("cpu")


def test_unknown_pipeline_is_rejected():
    with pytest.raises(ValueError, match="Unknown benchmark pipeline"):
        pipelines.build_recognizer("unknown", device="cpu")


@pytest.mark.parametrize(
    ("pipeline", "expected_builder"),
    [
        (pipelines.XLM_MINILM, pipelines._build_xlm_recognizer),
        (pipelines.OTTER_MINILM, pipelines._build_otter_recognizer),
    ],
)
def test_new_pipeline_selects_its_declared_recognizer(
    monkeypatch, pipeline, expected_builder
):
    assert pipelines._RECOGNIZER_BUILDERS[pipeline] is expected_builder

    recognizer = object()
    factory = Mock(return_value=recognizer)
    monkeypatch.setitem(pipelines._RECOGNIZER_BUILDERS, pipeline, factory)

    assert pipelines.build_recognizer(pipeline, device="cuda") is recognizer
    factory.assert_called_once_with("cuda")


def test_prior_uses_gliner2_for_recognition(monkeypatch, tmp_path):
    recognizer = SimpleNamespace(model=Mock())
    gliner_factory = Mock(return_value=recognizer)
    snapshot = tmp_path / "gliner-snapshot"
    _patch_module_classes(monkeypatch, GLiNER2Recognizer=gliner_factory)
    monkeypatch.setattr(pipelines, "_snapshot", lambda key: snapshot)

    assert pipelines.build_recognizer(pipelines.PRIOR, device="cpu") is recognizer
    gliner_factory.assert_called_once_with(model_name=str(snapshot))


def test_prior_uses_the_prior_resolver_on_the_upstream_model(monkeypatch):
    snapshot = Path("/cache/minilm-snapshot")
    resolver = SimpleNamespace(transformer=Mock(), reranker=None)
    prior_factory = Mock(return_value=resolver)
    _patch_module_classes(monkeypatch, PriorResolver=prior_factory)
    monkeypatch.setattr(pipelines, "GAZETTEER_NAME", "geonames")
    snapshot_factory = Mock(return_value=snapshot)
    monkeypatch.setattr(pipelines, "_snapshot", snapshot_factory)

    result = pipelines.build_resolver(
        pipelines.PRIOR, device="cuda", min_similarity=0.0
    )

    assert result is resolver
    snapshot_factory.assert_called_once_with("minilm")
    prior_factory.assert_called_once_with(
        model_name=str(snapshot),
        gazetteer_name="geonames",
        min_similarity=0.0,
        population_weight=0.3,
        inflection_fallback=False,
    )
    assert resolver.benchmark_model_pin == pipelines._pinned_name("minilm")
    resolver.transformer.to.assert_called_once_with("cuda")


# One factor at a time: every ablation is hybrid with the settings listed.
ABLATIONS = {
    "trim": (0.0, True),
    "population": (0.1, False),
    "population-0.05": (0.05, False),
    "population-0.2": (0.2, False),
    "population-0.3": (0.3, False),
    "population-0.5": (0.5, False),
    "population-1.0": (1.0, False),
}


def test_ablations_are_offered_but_not_run_by_default():
    """The ablation pipelines are selectable and stay out of a default run."""
    assert set(ABLATIONS) <= set(pipelines.PIPELINES)
    assert not set(ABLATIONS) & set(pipelines.DEFAULT_PIPELINES)
    assert build_parser().parse_args(["--pipeline", "trim"]).pipeline == ["trim"]


@pytest.mark.parametrize(("pipeline", "settings"), sorted(ABLATIONS.items()))
def test_each_ablation_changes_one_setting(monkeypatch, pipeline, settings):
    """Each builds the prior resolver with exactly its settings."""
    weight, fallback = settings
    resolver = SimpleNamespace(transformer=Mock(), reranker=None)
    prior_factory = Mock(return_value=resolver)
    _patch_module_classes(monkeypatch, PriorResolver=prior_factory)
    monkeypatch.setattr(pipelines, "GAZETTEER_NAME", "geonames")
    snapshot = Path("/cache/minilm-snapshot")
    snapshot_factory = Mock(return_value=snapshot)
    monkeypatch.setattr(pipelines, "_snapshot", snapshot_factory)

    pipelines.build_resolver(pipeline, device="cpu", min_similarity=0.0)

    snapshot_factory.assert_called_once_with("minilm")
    prior_factory.assert_called_once_with(
        model_name=str(snapshot),
        gazetteer_name="geonames",
        min_similarity=0.0,
        population_weight=weight,
        inflection_fallback=fallback,
    )


@pytest.mark.parametrize("pipeline", sorted(ABLATIONS))
def test_ablations_share_hybrids_recognizer(monkeypatch, pipeline):
    """Only resolution varies, so recognition is GLiNER2 throughout."""
    recognizer = SimpleNamespace(model=Mock())
    gliner_factory = Mock(return_value=recognizer)
    _patch_module_classes(monkeypatch, GLiNER2Recognizer=gliner_factory)
    monkeypatch.setattr(pipelines, "_snapshot", lambda key: Path("/cache/gliner"))

    assert pipelines.build_recognizer(pipeline, device="cpu") is recognizer


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", False, "cpu"),
        ("cuda", True, "cuda"),
    ],
)
def test_resolve_device_uses_availability_for_auto_and_explicit_devices(
    monkeypatch, requested, available, expected
):
    torch = ModuleType("torch")
    torch.__dict__["cuda"] = SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert pipelines.resolve_device(requested) == expected


def test_resolve_device_rejects_unavailable_explicit_cuda(monkeypatch):
    torch = ModuleType("torch")
    torch.__dict__["cuda"] = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        pipelines.resolve_device("cuda")


@pytest.mark.parametrize(
    ("device", "available", "gpu_name", "expected"),
    [
        ("cpu", True, "unused", "cpu"),
        ("cuda:0", False, "unused", "cuda:0"),
        ("cuda:0", True, "Test GPU", "cuda:0 (Test GPU)"),
    ],
)
def test_describe_device_reports_a_gpu_name_only_for_available_cuda(
    monkeypatch, device, available, gpu_name, expected
):
    torch = ModuleType("torch")
    torch.__dict__["cuda"] = SimpleNamespace(
        is_available=lambda: available,
        get_device_name=lambda index: gpu_name,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert pipelines.describe_device(device) == expected


@pytest.mark.parametrize(
    ("recognizer", "resolver", "expected"),
    [
        (None, None, {}),
        (SimpleNamespace(model_name="gliner"), None, {"recognizer": "gliner"}),
        (None, SimpleNamespace(model_name="jina"), {"resolver": "jina"}),
        (
            None,
            SimpleNamespace(model_name="jina", reranker_name="cross-encoder"),
            {"resolver": "jina", "reranker": "cross-encoder"},
        ),
    ],
)
def test_model_names_describe_the_loaded_recognizer_resolver_and_reranker(
    recognizer, resolver, expected
):
    assert pipelines.model_names(recognizer, resolver) == expected


def test_model_names_use_pins_for_newly_loaded_snapshots():
    recognizer = SimpleNamespace(
        model_name="/tmp/cache/snapshot",
        benchmark_model_pin="recognizer@revision",
    )
    resolver = SimpleNamespace(
        model_name="/tmp/minilm",
        benchmark_model_pin="resolver@revision",
        reranker_name="/tmp/reranker",
        benchmark_reranker_pin="reranker@revision",
    )

    assert pipelines.model_names(recognizer, resolver) == {
        "recognizer": "recognizer@revision",
        "resolver": "resolver@revision",
        "reranker": "reranker@revision",
    }


def test_new_pipeline_identities_pin_shared_resolution_models():
    xlm_pins = pipelines.pipeline_model_pins(pipelines.XLM_MINILM)
    otter_pins = pipelines.pipeline_model_pins(pipelines.OTTER_MINILM)

    assert xlm_pins["recognizer"] != otter_pins["recognizer"]
    assert xlm_pins["resolver"] == otter_pins["resolver"]
    assert (
        xlm_pins["context_sentence_splitter"] == otter_pins["context_sentence_splitter"]
    )


@pytest.mark.parametrize(
    "pipeline",
    [pipelines.UPSTREAM, pipelines.HYBRID, pipelines.PRIOR, *pipelines.ABLATIONS],
)
def test_existing_minilm_pipeline_identities_pin_the_frozen_revision(pipeline):
    assert pipelines.pipeline_model_pins(pipeline)["resolver"] == (
        pipelines._pinned_name("minilm")
    )


def test_otter_integration_provenance_matches_retained_checkpoint_plan():
    provenance = pipelines.otter_integration_provenance()

    assert {
        key: provenance[key]
        for key in (
            "model",
            "model_sha256",
            "source_manifest_sha256",
            "source_review_sha256",
            "plan_status",
            "language_support",
        )
    } == {
        "model": "whoisjones/otter-cross-mmbert@8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d",
        "model_sha256": "8987080bfc3e6672a75fb19ffe904d39e79ad804eabfda8247a62c347fb024b2",
        "source_manifest_sha256": "2d6e72bacb2e933cd27d0f5bd7ef53ebac3be0eebcbbe6cfbf36f22014ba2d78",
        "source_review_sha256": "458e5cae6d0ce62b0d7a7fdb91a7617b1092463ce62a14471d86974552fcab75",
        "plan_status": "planned",
        "language_support": "unspecified",
    }
    assert isinstance(provenance["plan_lock_matches_current"], bool)


def test_otter_checkpoint_config_matches_the_retained_arm():
    config = SimpleNamespace(
        architecture="cross_encoder",
        max_seq_length=1024,
        max_span_length=30,
    )
    pipelines._verify_otter_model_config(SimpleNamespace(config=config))


def test_otter_checkpoint_config_must_match_the_retained_arm():
    model = SimpleNamespace(
        config=SimpleNamespace(
            architecture="bi_encoder",
            max_seq_length=1024,
            max_span_length=30,
        )
    )

    with pytest.raises(RuntimeError, match="differs from its retained arm plan"):
        pipelines._verify_otter_model_config(model)


def test_snapshot_download_is_pinned_to_the_requested_revision(monkeypatch, tmp_path):
    revision = pipelines.missing_matrix.MODEL_PINS["minilm"][1]
    snapshot = tmp_path / revision
    hub = ModuleType("huggingface_hub")
    download = Mock(return_value=str(snapshot))
    hub.__dict__["snapshot_download"] = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))

    assert pipelines._snapshot("minilm") == snapshot
    download.assert_called_once_with(
        repo_id=pipelines.missing_matrix.MODEL_PINS["minilm"][0],
        revision=revision,
        cache_dir=str(tmp_path / "cache" / "hub"),
    )


def test_snapshot_download_rejects_a_different_revision(monkeypatch, tmp_path):
    hub = ModuleType("huggingface_hub")
    hub.__dict__["snapshot_download"] = lambda **_kwargs: str(tmp_path / "mutable")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    with pytest.raises(RuntimeError, match="expected"):
        pipelines._snapshot("minilm")


@pytest.mark.parametrize(
    ("installed", "expected_error"),
    [(pipelines.SPACY_CONTEXT_MODEL_VERSION, None), ("1.0.0", "Expected")],
)
def test_context_model_version_is_checked(monkeypatch, installed, expected_error):
    monkeypatch.setattr(
        pipelines.importlib.metadata, "version", lambda _name: installed
    )

    if expected_error:
        with pytest.raises(RuntimeError, match=expected_error):
            pipelines._require_context_model()
    else:
        pipelines._require_context_model()


def test_context_model_must_be_installed(monkeypatch):
    def missing(_name):
        raise pipelines.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(pipelines.importlib.metadata, "version", missing)

    with pytest.raises(RuntimeError, match="Install the locked"):
        pipelines._require_context_model()


def test_xlm_recognizer_build_and_prediction_use_pinned_snapshot(monkeypatch, tmp_path):
    class FakeRecognizer:
        def __init__(self, model_name):
            self.model_name = model_name

    class FakeModel:
        def eval(self):
            self.evaluated = True

    tokenizer = object()
    model = FakeModel()
    predictions = [
        [
            {"entity_group": "LOC", "start": 4, "end": 8},
            {"entity_group": "B-LOC", "start": 0, "end": 3},
            {"entity_group": "PER", "start": 9, "end": 12},
            {"entity_group": "LOC", "start": None, "end": 12},
        ]
    ]
    inference = Mock(return_value=predictions)
    tokenizer_factory = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    model_factory = SimpleNamespace(from_pretrained=Mock(return_value=model))
    transformers = ModuleType("transformers")
    transformers.__dict__.update(
        {
            "AutoTokenizer": tokenizer_factory,
            "AutoModelForTokenClassification": model_factory,
            "pipeline": Mock(return_value=inference),
        }
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    recognizers = ModuleType("geoparser.modules.recognizers")
    recognizers.__dict__["Recognizer"] = FakeRecognizer
    monkeypatch.setitem(sys.modules, "geoparser.modules.recognizers", recognizers)
    snapshot = tmp_path / pipelines.missing_matrix.MODEL_PINS["xlm_r"][1]
    monkeypatch.setattr(pipelines, "_snapshot", lambda _key: snapshot)

    recognizer = pipelines._build_xlm_recognizer("cuda")

    assert recognizer.predict(["Zurich"]) == [[(0, 3), (4, 8)]]
    assert recognizer.benchmark_model_pin == pipelines._pinned_name("xlm_r")
    assert model.evaluated
    tokenizer_factory.from_pretrained.assert_called_once_with(
        str(snapshot), use_fast=True, local_files_only=True
    )
    model_factory.from_pretrained.assert_called_once_with(
        str(snapshot), local_files_only=True
    )
    transformers.pipeline.assert_called_once()


def test_xlm_prediction_rejects_a_changed_document_count():
    with pytest.raises(RuntimeError, match="different number"):
        pipelines._predict_xlm_spans(lambda *_args, **_kwargs: [], ["Zurich"])


def test_otter_snapshot_verifies_artifact_sizes_hashes_and_import_closure(
    monkeypatch, tmp_path
):
    artifact_bytes = b"verified tiny model source"
    (tmp_path / "model.safetensors").write_bytes(artifact_bytes)
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    artifact = {
        "path": "model.safetensors",
        "bytes": len(artifact_bytes),
        "sha256": digest,
    }
    manifest = {"inference_imports": ["model.safetensors"]}
    entry = {"files": [artifact]}
    monkeypatch.setattr(pipelines, "_otter_arm_plan", Mock())
    monkeypatch.setattr(pipelines, "_otter_source_entry", lambda: (manifest, entry))

    pipelines._verify_otter_snapshot(tmp_path)


def test_otter_snapshot_rejects_artifact_hash_mismatch(monkeypatch, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"wrong")
    monkeypatch.setattr(pipelines, "_otter_arm_plan", Mock())
    monkeypatch.setattr(
        pipelines,
        "_otter_source_entry",
        lambda: (
            {"inference_imports": ["model.safetensors"]},
            {"files": [{"path": "model.safetensors", "bytes": 5, "sha256": "wrong"}]},
        ),
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        pipelines._verify_otter_snapshot(tmp_path)
