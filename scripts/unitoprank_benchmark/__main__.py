"""Check a local UniTopRank checkout against the reviewed pin, offline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scripts.unitoprank_benchmark.pins import (
    COMMIT,
    LICENSE_SPDX,
    REPOSITORY,
    REVIEWED_BLOBS,
    verify_checkout,
)


def build_parser() -> argparse.ArgumentParser:
    """Describe the command line; building it reads and writes nothing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True, metavar="DIR")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the pin and any mismatch; never import the code or open a socket."""
    arguments = build_parser().parse_args(argv)
    problems = verify_checkout(arguments.checkout)
    print(
        json.dumps(
            {
                "repository": REPOSITORY,
                "commit": COMMIT,
                "license": LICENSE_SPDX,
                "files_pinned": len(REVIEWED_BLOBS),
                "problems": problems,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
