"""Validate UNER configuration metadata and optional local source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.uner_benchmark.inventory import (
    Dataset,
    Split,
    inventory_summary,
    load_local,
    read_manifest,
    validate_configurations,
)


def build_parser() -> argparse.ArgumentParser:
    """Expose a dry-run command with no network or model execution mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--configuration", action="append")
    parser.add_argument("--cache-dir", type=Path)
    return parser


def _select(specs: tuple[Dataset, ...], names: list[str] | None) -> tuple[Dataset, ...]:
    """Select exact configurations without silently discarding invalid names."""
    validate_configurations(specs)
    if names is None:
        return specs
    if len(set(names)) != len(names):
        message = "Duplicate configuration selection"
        raise ValueError(message)
    registry = _registry(specs)
    unknown = set(names) - registry.keys()
    if unknown:
        message = f"Unknown configurations: {sorted(unknown)}"
        raise ValueError(message)
    return tuple(registry[name] for name in names)


def _registry(specs: tuple[Dataset, ...]) -> dict[str, Dataset]:
    """Index explicitly validated source configurations."""
    return {spec.configuration: spec for spec in specs}


def _local_source(spec: Dataset, split: Split, cache: Path) -> dict[str, Any]:
    """Retain a failed source explicitly, or report complete validated counts."""
    base = {"configuration": spec.configuration, "split": split}
    try:
        loaded = load_local(spec, split, cache)
    except (OSError, ValueError) as error:
        return {
            **base,
            "status": "failed",
            "error_type": type(error).__name__,
            "reason": str(error),
        }
    corpus = loaded.corpus
    return {
        **base,
        "status": "valid",
        "sha256": loaded.sha256,
        "sentences": corpus.sentence_count,
        "documents": corpus.document_count,
        "tokens": corpus.token_count,
        "locations": corpus.location_count,
        "empty_texts": corpus.empty_text_count,
    }


def _report(
    specs: tuple[Dataset, ...], split: Split, cache: Path | None
) -> dict[str, Any]:
    """Distinguish metadata validation from an actual local corpus inspection."""
    report = inventory_summary(specs, split)
    report["model_evaluation"] = "not_run"
    report["validation"] = "registry_only" if cache is None else "local_sources"
    report["local_sources"] = []
    if cache is not None:
        report["local_sources"] = [
            _local_source(spec, split, cache)
            for spec in specs
            if spec.status(split) == "available"
        ]
    return report


def main(argv: list[str] | None = None) -> int:
    """Print the full dry-run inventory; return two for missing or invalid data."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        specs = _select(read_manifest(), args.configuration)
        report = _report(specs, args.split, args.cache_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failed = any(source["status"] == "failed" for source in report["local_sources"])
    return 2 if failed or not report["selected_configurations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
