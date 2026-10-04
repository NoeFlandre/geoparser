"""Run exact-span scoring and record reproducibility metadata."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from scripts import git_provenance
from scripts.panx_benchmark.checkpoint import (
    ModelLanguageCheckpoints,
    require_clean_commit,
)
from scripts.panx_benchmark.constants import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    GLINER_THRESHOLD,
    MODELS,
    SEED,
    ModelSpec,
)
from scripts.panx_benchmark.data import (
    TARGET_LANGUAGES_PATH,
    TEST_SPLITS_PATH,
    Example,
    LoadedDataset,
    split_manifest,
    target_languages,
)
from scripts.panx_benchmark.metrics import Counts, macro_scores
from scripts.panx_benchmark.models import BatchPredictor, LoadedModel, load_model


def configure_cpu(seed: int = SEED) -> int:
    """Seed model runtimes and cap CPU threads to this process's quota."""
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    available_cpus = _cgroup_cpu_count() or (os.cpu_count() or 1)
    thread_count = max(1, int(available_cpus))
    torch.set_num_threads(thread_count)
    return thread_count


def _cgroup_cpu_count() -> float | None:
    """Read a Linux cgroup CPU quota when the environment provides one."""
    quota_path = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota, period = quota_path.read_text(encoding="utf-8").split()
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    try:
        return max(1.0, int(quota) / int(period))
    except (ValueError, ZeroDivisionError):
        return None


def _memory_bytes() -> int | None:
    """Read host memory total on Linux for the provenance record."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def hardware_facts(torch_threads: int) -> dict[str, Any]:
    """Record host, CPU allocation, memory and accelerator facts."""
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "cgroup_cpu_quota": _cgroup_cpu_count(),
        "torch_threads": torch_threads,
        "memory_total_bytes": _memory_bytes(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "device_used": "cpu",
    }


def _batches(
    examples: Sequence[Example], batch_size: int
) -> Iterator[Sequence[Example]]:
    """Partition sentences using the fixed, recorded model batch size."""
    for offset in range(0, len(examples), batch_size):
        yield examples[offset : offset + batch_size]


def _valid_predictions(
    predictor: BatchPredictor,
    batch: Sequence[Example],
) -> list[set[tuple[int, int]]]:
    """Run one inference batch and reject missing sentence predictions."""
    predictions = predictor.predict_batch([example.text for example in batch])
    _validate_prediction_count(predictions, len(batch))
    return predictions


def _validate_prediction_count(
    predictions: list[set[tuple[int, int]]], batch_size: int
) -> None:
    """Require one prediction collection for each submitted sentence."""
    if len(predictions) != batch_size:
        message = "Recognizer returned a different number of predictions"
        raise ValueError(message)


def _warm_up(
    predictor: BatchPredictor,
    examples_by_language: dict[str, tuple[Example, ...]],
    supported_languages: tuple[str, ...] | None,
    batch_size: int,
) -> tuple[float, int]:
    """Warm one fixed batch before timing steady-state test inference."""
    warmup_language = _first_warmup_language(examples_by_language, supported_languages)
    if warmup_language is None:
        return 0.0, 0
    examples = examples_by_language[warmup_language][:batch_size]
    started = time.perf_counter()
    _valid_predictions(predictor, examples)
    return time.perf_counter() - started, len(examples)


def _first_warmup_language(
    examples_by_language: dict[str, tuple[Example, ...]],
    supported_languages: tuple[str, ...] | None,
) -> str | None:
    """Find the first nonempty language a model is allowed to evaluate."""
    for language, examples in examples_by_language.items():
        if examples and (
            supported_languages is None or language in supported_languages
        ):
            return language
    return None


def _support_status(spec: ModelSpec, language: str) -> str:
    """Describe documented support separately from transfer evaluation."""
    if spec.key == "spacy_en":
        return "documented" if language == "en" else "not_evaluated_english_only"
    if spec.documented_languages is None:
        return "evaluated_multilingual_claim"
    if language in spec.documented_languages:
        return "fine_tuned_language"
    return "cross_lingual_transfer"


def _score_language(
    predictor: BatchPredictor,
    examples: tuple[Example, ...],
    batch_size: int,
) -> Counts:
    """Score every sentence in one language and time only model inference."""
    counts = Counts()
    for batch in _batches(examples, batch_size):
        started = time.perf_counter()
        predictions = _valid_predictions(predictor, batch)
        counts.elapsed_seconds += time.perf_counter() - started
        for example, spans in zip(batch, predictions, strict=True):
            counts.add(example.gold_spans, spans, text_length=len(example.text))
            counts.malformed_gold_tags += example.malformed_location_tags
    return counts


def _micro_scores(
    per_language: dict[str, dict[str, float | int]],
) -> dict[str, float | int]:
    """Aggregate mention-level counts across the languages a model evaluated."""
    total = Counts()
    for row in per_language.values():
        total.true_positive += int(row["true_positive"])
        total.false_positive += int(row["false_positive"])
        total.false_negative += int(row["false_negative"])
        total.sentences += int(row["sentences"])
        total.malformed_gold_tags += int(row.get("malformed_gold_tags", 0))
        total.invalid_prediction_spans += int(row.get("invalid_prediction_spans", 0))
        total.elapsed_seconds += float(row["elapsed_seconds"])
    return total.scores()


def _file_sha256(path: Path) -> str:
    """Hash a pinned input manifest into the immutable run identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_fingerprints(dataset: LoadedDataset) -> dict[str, dict[str, Any]]:
    """Fingerprint the ordered held-out text and gold spans for every language."""
    fingerprints = {}
    for language, examples in dataset.examples_by_language.items():
        digest = hashlib.sha256()
        for example in examples:
            record = {
                "language": example.language,
                "text": example.text,
                "gold_spans": sorted(example.gold_spans),
                "malformed_location_tags": example.malformed_location_tags,
            }
            digest.update(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
        fingerprints[language] = {
            "example_count": len(examples),
            "sha256": digest.hexdigest(),
        }
    return fingerprints


def _checkpoint_identity(
    dataset: LoadedDataset,
    models: Sequence[ModelSpec],
    repository_commit: str,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    """Describe every code, data, model and runtime input that shapes scores."""
    manifest = split_manifest()
    return {
        "repository_commit": repository_commit,
        "input_manifests": {
            "target_languages_sha256": _file_sha256(TARGET_LANGUAGES_PATH),
            "test_split_sha256": _file_sha256(TEST_SPLITS_PATH),
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "canonical_target_languages": list(target_languages()),
            "eligible_languages": list(dataset.examples_by_language),
            "missing_target_languages": manifest["missing_target_languages"],
            "source_test_examples_by_language": dataset.source_counts,
            "held_out_examples_by_language": _dataset_fingerprints(dataset),
            "limit_per_language": dataset.limit_per_language,
        },
        "evaluation": {
            "seed": SEED,
            "batch_sizes_by_model": {spec.key: spec.batch_size for spec in models},
            "gliner_threshold": GLINER_THRESHOLD,
            "device": "cpu",
            "span_policy": "exact half-open Python character offsets",
        },
        "models": [
            {
                "key": spec.key,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "batch_size": spec.batch_size,
                "documented_languages": list(spec.documented_languages or ()),
            }
            for spec in models
        ],
        "hardware": hardware,
    }


def _language_result(
    spec: ModelSpec,
    predictor: BatchPredictor,
    language: str,
    examples: tuple[Example, ...],
    source_example_count: int,
) -> dict[str, Any]:
    """Score one language or record why it was deliberately not evaluated."""
    support = _support_status(spec, language)
    if support == "not_evaluated_english_only":
        return {
            "status": support,
            "documented_support": False,
            "source_test_examples": source_example_count,
            "evaluated_examples": 0,
            "metrics": None,
        }
    counts = _score_language(predictor, examples, spec.batch_size)
    return {
        "status": "evaluated",
        "documented_support": support,
        "source_test_examples": source_example_count,
        "evaluated_examples": len(examples),
        "metrics": counts.scores(),
    }


def _language_results(
    spec: ModelSpec,
    predictor: BatchPredictor,
    dataset: LoadedDataset,
    checkpoints: ModelLanguageCheckpoints | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the per-language records for one recognizer."""
    results = {}
    for language, examples in dataset.examples_by_language.items():
        source_count = dataset.source_counts[language]
        result = (
            checkpoints.load_language(
                spec.key,
                spec.model_id,
                spec.revision,
                language,
                source_count,
            )
            if checkpoints is not None
            else None
        )
        if result is None:
            result = _language_result(spec, predictor, language, examples, source_count)
            if checkpoints is not None:
                checkpoints.save_language(
                    spec.key,
                    spec.model_id,
                    spec.revision,
                    language,
                    source_count,
                    result,
                )
        results[language] = result
    return results


def _metric_rows(
    per_language: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Select evaluated language metrics for aggregate score calculations."""
    return {
        language: row["metrics"]
        for language, row in per_language.items()
        if row["metrics"] is not None
    }


def _evaluated_example_count(per_language: dict[str, dict[str, Any]]) -> int:
    """Sum sentences actually passed through a recognizer."""
    return sum(int(row["evaluated_examples"]) for row in per_language.values())


def _steady_inference_summary(
    spec: ModelSpec,
    dataset: LoadedDataset,
    per_language: dict[str, dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, float | int | None]:
    """Summarize measured throughput and a linear full-matrix estimate."""
    evaluated_examples = _evaluated_example_count(per_language)
    examples_per_second = (
        evaluated_examples / elapsed_seconds if elapsed_seconds else 0.0
    )
    full_matrix_examples = (
        dataset.source_counts.get("en", 0)
        if spec.key == "spacy_en"
        else sum(dataset.source_counts.values())
    )
    full_matrix_seconds = (
        full_matrix_examples / examples_per_second if examples_per_second else None
    )
    return {
        "evaluated_examples": evaluated_examples,
        "steady_examples_per_second": examples_per_second,
        "full_matrix_estimated_inference_seconds": full_matrix_seconds,
    }


def evaluate_model(
    spec: ModelSpec,
    loaded: LoadedModel,
    dataset: LoadedDataset,
    checkpoints: ModelLanguageCheckpoints | None = None,
) -> dict[str, Any]:
    """Evaluate one recognizer and separate startup, warmup and steady timing."""
    predictor = loaded.predictor
    warmup_seconds, warmup_examples = _warm_up(
        predictor,
        dataset.examples_by_language,
        spec.documented_languages,
        spec.batch_size,
    )
    per_language = _language_results(spec, predictor, dataset, checkpoints)
    inference_seconds = sum(
        float(row["metrics"]["elapsed_seconds"])
        for row in per_language.values()
        if row["metrics"] is not None
    )
    evaluated = _metric_rows(per_language)
    inference_summary = _steady_inference_summary(
        spec, dataset, per_language, inference_seconds
    )
    return {
        "key": spec.key,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "model_card_url": spec.model_card_url,
        "coverage_note": spec.coverage_note,
        "documented_languages": list(spec.documented_languages or ()),
        "training_data_note": spec.training_data_note,
        "training_overlap_note": spec.overlap_note,
        "location_mapping": _location_mapping(spec),
        "batch_size": spec.batch_size,
        "gliner_threshold": GLINER_THRESHOLD if spec.key == "gliner2_multi" else None,
        "device": "cpu",
        "hub_snapshot_cached_before_run": loaded.cache_hit,
        "checkpoint_path": loaded.checkpoint_path,
        "checkpoint_download_seconds": loaded.download_seconds,
        "model_load_seconds": loaded.load_seconds,
        "warmup_seconds": warmup_seconds,
        "warmup_examples": warmup_examples,
        "steady_inference_seconds": inference_seconds,
        **inference_summary,
        "macro": macro_scores(evaluated),
        "micro": _micro_scores(evaluated),
        "per_language": per_language,
    }


def _location_mapping(spec: ModelSpec) -> str:
    """Return the checkpoint-to-WikiANN mapping used in this run."""
    if spec.key == "spacy_en":
        return "FAC, GPE, LOC -> WikiANN LOC"
    if spec.key == "gliner2_multi":
        return "city, country, location -> WikiANN LOC; exact spans deduplicated"
    return "B-LOC/I-LOC -> WikiANN LOC; token labels grouped to character spans"


def _elapsed_full_matrix_seconds(models: list[dict[str, Any]]) -> float | None:
    """Sum linear full-split inference estimates from the bounded run."""
    estimates = [model["full_matrix_estimated_inference_seconds"] for model in models]
    if any(estimate is None for estimate in estimates):
        return None
    return sum(float(estimate) for estimate in estimates)


def _commit_id() -> str:
    """Return the exact source commit, with dirty state called out."""
    return git_provenance.commit_id(short=False, cwd=None)


def _evaluation_scope_metadata(limit: int | None) -> dict[str, str | bool]:
    """Describe whether these scores cover the held-out split or a sample."""
    if limit is None:
        return {
            "evaluation_kind": "full_test_split",
            "full_quality_comparison": True,
            "full_matrix_estimate_note": (
                "Measured inference over the complete pinned test intersection."
            ),
        }
    return {
        "evaluation_kind": "bounded_feasibility_sample",
        "full_quality_comparison": False,
        "full_matrix_estimate_note": (
            "Linear estimate from measured steady-state inference throughput; "
            "excludes model download/load and data acquisition. Small feasibility "
            "samples are not a quality result and may not predict every language."
        ),
    }


def run_benchmark(
    dataset: LoadedDataset,
    *,
    cache_dir: Path,
    thread_count: int,
    models_to_run: Sequence[ModelSpec] = MODELS,
    checkpoint_dir: Path | None = None,
    repository_commit: str | None = None,
) -> dict[str, Any]:
    """Run and report the three fixed model comparisons without training."""
    started_at = time.time()
    repository_commit = require_clean_commit(repository_commit or _commit_id())
    hardware = hardware_facts(thread_count)
    checkpoints = ModelLanguageCheckpoints(
        checkpoint_dir or cache_dir / "benchmark-checkpoints",
        _checkpoint_identity(dataset, models_to_run, repository_commit, hardware),
    )
    models: list[dict[str, Any]] = []
    for spec in models_to_run:
        loaded = load_model(spec, cache_dir)
        models.append(
            evaluate_model(
                spec,
                loaded,
                dataset,
                checkpoints,
            )
        )
        del loaded
        gc.collect()

    manifest = split_manifest()
    limit = dataset.limit_per_language
    return {
        "benchmark": "PAN-X/WikiANN location recognition (#98)",
        **_evaluation_scope_metadata(limit),
        "seed": SEED,
        "repository_commit": repository_commit,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "hardware": hardware,
        "checkpoint_snapshot_id": checkpoints.snapshot_id,
        "checkpoint_directory": str(checkpoints.directory),
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "canonical_target_count": len(manifest["eligible_languages"])
            + len(manifest["missing_target_languages"]),
            "eligible_languages": manifest["eligible_languages"],
            "missing_target_languages": manifest["missing_target_languages"],
            "source_test_examples": dataset.source_example_count,
            "evaluated_examples": dataset.evaluated_example_count,
            "source_test_examples_by_language": dataset.source_counts,
            "data_load_seconds": dataset.load_seconds,
            "limit_per_language": limit,
        },
        "evaluation": {
            "batch_sizes_by_model": {
                spec.key: spec.batch_size for spec in models_to_run
            },
            "seed": SEED,
            "span_policy": "exact half-open Python character offsets",
            "gold_text_reconstruction": "WikiANN tokens joined by one ASCII space",
            "zero_division": 0,
            "training_or_finetuning": False,
        },
        "models": models,
        "estimated_full_matrix_inference_seconds": _elapsed_full_matrix_seconds(models),
    }
