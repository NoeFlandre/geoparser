"""Validate a local experiment inventory without downloads or compute jobs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from scripts.benchmark_protocol.schema import Experiment
from scripts.benchmark_protocol.summary import inventory_summary


def main(argv: Sequence[str] | None = None) -> int:
    """Print schema or a dry-run inventory; never load a dataset or a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--dry-run", type=Path, metavar="MANIFEST.json")
    operation.add_argument("--schema", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.schema:
        print(json.dumps(Experiment.model_json_schema(), indent=2))
        return 0
    try:
        experiment = Experiment.model_validate_json(
            arguments.dry_run.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as error:
        print(f"Invalid benchmark manifest: {error}", file=sys.stderr)
        return 2
    print(json.dumps(inventory_summary(experiment), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
