"""Run the pinned WikiANN/PAN-X location-recognition benchmark.

python -m scripts.panx_benchmark --limit-per-language 8
python -m scripts.panx_benchmark --output-dir benchmark-evidence/panx/full
python -m scripts.panx_benchmark --model spacy_en --spacy-cross-lingual-transfer
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.panx_benchmark.checkpoint import require_clean_commit
from scripts.panx_benchmark.constants import MODELS, ModelSpec
from scripts.panx_benchmark.data import (
    TARGET_LANGUAGES_PATH,
    load_test_examples,
    read_json,
)
from scripts.panx_benchmark.report import write_reports
from scripts.panx_benchmark.runner import (
    BenchmarkRunOptions,
    _commit_id,
    configure_cpu,
    run_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the benchmark CLI options without loading models or datasets."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.panx_benchmark",
        description=(
            "Compare the pinned place recognizers on WikiANN test splits. "
            "Without a limit this evaluates all 423,100 test examples."
        ),
    )
    parser.add_argument(
        "--limit-per-language",
        type=int,
        help="Use only the first N test examples per eligible language for feasibility",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "GEOPARSER_PANX_CACHE",
                str(Path(tempfile.gettempdir()) / "geoparser-panx-cache"),
            )
        ),
        help="External cache for pinned Hub models and WikiANN test data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Base directory for immutable run identity and per-language checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Local report directory; defaults to a unique timestamped evidence path",
    )
    parser.add_argument(
        "--model",
        dest="model_keys",
        action="append",
        choices=tuple(spec.key for spec in MODELS),
        help=(
            "Run only the selected model key; may be repeated. By default all "
            "three pinned models run."
        ),
    )
    parser.add_argument(
        "--spacy-cross-lingual-transfer",
        action="store_true",
        help=(
            "Evaluate the pinned English spaCy model on every eligible language "
            "as cross-lingual transfer, not native multilingual support."
        ),
    )
    return parser


def _default_output_dir() -> Path:
    """Choose a unique local evidence directory for this invocation."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmark-evidence") / "panx" / timestamp


def _resource_facts(cache_dir: Path) -> dict[str, int | str]:
    """Record available storage without imposing an arbitrary disk threshold."""
    usage = shutil.disk_usage(cache_dir)
    return {
        "cache_directory": str(cache_dir.resolve()),
        "cache_free_bytes_before_run": usage.free,
        "cache_total_bytes": usage.total,
    }


def _validate_limit(parser: argparse.ArgumentParser, limit: int | None) -> None:
    """Reject a nonpositive feasibility sample before loading data."""
    if limit is not None and limit < 1:
        parser.error("--limit-per-language must be positive")


def _has_spacy_model(models: tuple[ModelSpec, ...]) -> bool:
    """Return whether spaCy English is among the selected recognizers."""
    return any(spec.key == "spacy_en" for spec in models)


def _selected_models(
    parser: argparse.ArgumentParser, model_keys: list[str] | None
) -> tuple[ModelSpec, ...]:
    """Resolve model keys and reject duplicate selections."""
    if not model_keys:
        return MODELS
    selected_keys = set(model_keys)
    if len(selected_keys) != len(model_keys):
        parser.error("--model keys must not be repeated")
    return tuple(spec for spec in MODELS if spec.key in selected_keys)


def _run_options(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> BenchmarkRunOptions:
    """Resolve model selection and validate the transfer-mode combination."""
    selected_models = _selected_models(parser, arguments.model_keys)
    if arguments.spacy_cross_lingual_transfer and not _has_spacy_model(selected_models):
        parser.error(
            "--spacy-cross-lingual-transfer requires --model spacy_en or the default model set"
        )
    return BenchmarkRunOptions(
        models_to_run=selected_models,
        spacy_cross_lingual_transfer=arguments.spacy_cross_lingual_transfer,
    )


def _checkpoint_directory(arguments: argparse.Namespace) -> Path:
    """Resolve explicit or cache-local checkpoint storage."""
    if arguments.checkpoint_dir is not None:
        return arguments.checkpoint_dir
    return arguments.cache_dir / "benchmark-checkpoints"


def _output_directory(arguments: argparse.Namespace) -> Path:
    """Resolve explicit or timestamped report output storage."""
    if arguments.output_dir is not None:
        return arguments.output_dir
    return _default_output_dir()


def main() -> int:
    """Load test data, run the fixed CPU matrix and save local reports."""
    parser = build_parser()
    arguments = parser.parse_args()
    _validate_limit(parser, arguments.limit_per_language)
    options = _run_options(parser, arguments)
    repository_commit = require_clean_commit(_commit_id())
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)
    resources = _resource_facts(arguments.cache_dir)
    thread_count = configure_cpu()
    dataset = load_test_examples(
        arguments.cache_dir,
        limit_per_language=arguments.limit_per_language,
    )
    result = run_benchmark(
        dataset,
        cache_dir=arguments.cache_dir,
        thread_count=thread_count,
        options=options,
        checkpoint_dir=_checkpoint_directory(arguments),
        repository_commit=repository_commit,
    )
    result["resources"] = resources
    result["language_list"] = read_json(TARGET_LANGUAGES_PATH)
    output_dir = _output_directory(arguments)
    json_path, markdown_path = write_reports(output_dir, result)
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
