"""Inventory and verify the pinned MasakhaNER 2.0 inputs.

python -m scripts.masakhaner_benchmark inventory
python -m scripts.masakhaner_benchmark verify --data-dir DIR

Neither command downloads anything or runs a model. The inventory is written to
benchmark-evidence/masakhaner/ by default. The verify command checks local files
against their pinned git blob ids.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scripts.masakhaner_benchmark.manifest import read_manifest
from scripts.masakhaner_benchmark.pins import OK, check_directory
from scripts.masakhaner_benchmark.report import write_report


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser without reading any data."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.masakhaner_benchmark",
        description="Inventory and verify the pinned MasakhaNER 2.0 benchmark inputs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser(
        "inventory", help="Write the covered and missing language report"
    )
    inventory.add_argument(
        "--output-dir",
        type=Path,
        help="Report directory; defaults to benchmark-evidence/masakhaner/inventory-<date>",
    )
    verify = commands.add_parser(
        "verify", help="Check local split files against their pinned blob ids"
    )
    verify.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Local copy of MasakhaNER2.0/data, with one folder per configuration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return a process exit status."""
    arguments = build_parser().parse_args(argv)
    manifest = read_manifest()
    if arguments.command == "inventory":
        output_dir = arguments.output_dir or (
            Path("benchmark-evidence")
            / "masakhaner"
            / f"inventory-{manifest['retrieved_on']}"
        )
        json_path, markdown_path = write_report(output_dir, manifest)
        print(f"Wrote {markdown_path}")
        print(f"Wrote {json_path}")
        return 0
    rows = check_directory(arguments.data_dir, manifest["languages"])
    for row in rows:
        print(f"{row['status']:<8} {row['config']} {row['split']} {row['local_path']}")
    return 0 if all(row["status"] == OK for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
