"""Run the pinned WikiANN/PAN-X location-recognition benchmark.

python -m scripts.panx_benchmark --limit-per-language 8
python -m scripts.panx_benchmark --output-dir benchmark-evidence/panx/full
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.panx_benchmark.checkpoint import require_clean_commit
from scripts.panx_benchmark.constants import MODELS
from scripts.panx_benchmark.data import (
    TARGET_LANGUAGES_PATH,
    load_test_examples,
    read_json,
)
from scripts.panx_benchmark.report import write_reports
from scripts.panx_benchmark.runner import _commit_id, configure_cpu, run_benchmark


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


def main() -> int:
    """Load test data, run the fixed CPU matrix and save local reports."""
    arguments = build_parser().parse_args()
    if arguments.limit_per_language is not None and arguments.limit_per_language < 1:
        build_parser().error("--limit-per-language must be positive")
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
        models_to_run=MODELS,
        checkpoint_dir=(
            arguments.checkpoint_dir or arguments.cache_dir / "benchmark-checkpoints"
        ),
        repository_commit=repository_commit,
    )
    result["resources"] = resources
    result["language_list"] = read_json(TARGET_LANGUAGES_PATH)
    output_dir = arguments.output_dir or _default_output_dir()
    json_path, markdown_path = write_reports(output_dir, result)
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
