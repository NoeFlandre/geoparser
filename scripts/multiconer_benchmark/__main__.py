"""Dry-run the pinned inventory or validate one local split file. No downloads."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts._cli import EXIT_ERROR, EXIT_OK
from scripts.multiconer_benchmark.conll import ConllSummary, summarize_conll
from scripts.multiconer_benchmark.manifest import (
    DEFAULT_MANIFEST_PATH,
    check_local_file,
    intersection_languages,
    inventory_report,
    load_manifest,
)


def _parse_local(data: bytes, language: str | None) -> ConllSummary:
    """Count the records of verified UTF-8 bytes, expecting the given language."""
    if language not in intersection_languages(load_manifest()):
        message = f"{language!r} is not a dataset language in the intersection"
        raise ValueError(message)
    # Match the universal newlines that reading the file as text applied before.
    text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return summarize_conll(text, expected_domain=language)


def build_parser() -> argparse.ArgumentParser:
    """Describe the command line; building it reads and writes nothing."""
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument(
        "--manifest",
        nargs="?",
        const=DEFAULT_MANIFEST_PATH,
        type=Path,
        metavar="PATH",
        help="print the pinned inventory",
    )
    operation.add_argument(
        "--validate-conll",
        type=Path,
        metavar="FILE",
        help="parse one local split file and count invalid records",
    )
    parser.add_argument("--language", help="dataset language code for --validate-conll")
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        help="pinned split the file belongs to, for --validate-conll",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the pinned inventory, or validate one local file and report it."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.manifest is not None:
        return _print_manifest(arguments.manifest)
    if arguments.language is None or arguments.split is None:
        parser.error("--validate-conll needs --language and --split")
    return _validate_file(arguments.validate_conll, arguments.language, arguments.split)


def _print_manifest(path: Path) -> int:
    """Print the inventory for a pinned manifest, or fail with the reason."""
    try:
        manifest = load_manifest(path)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        print(f"Invalid MultiCoNER manifest: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(inventory_report(manifest), indent=2, ensure_ascii=False))
    return EXIT_OK


def _validate_file(path: Path, language: str, split: str) -> int:
    """Check a file against its pinned split, then print its record counts."""
    try:
        data = check_local_file(path, language, split, load_manifest())
        summary = _parse_local(data, language)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Invalid MultiCoNER file: {error}", file=sys.stderr)
        return EXIT_ERROR
    report = {
        "file": path.name,
        "language": language,
        "split": split,
        "records": summary.records,
        "valid": summary.valid,
        "invalid": summary.invalid,
        "invalid_reasons": summary.invalid_reasons,
        "location_spans": summary.location_spans,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return EXIT_ERROR if summary.invalid else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
