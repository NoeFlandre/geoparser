"""Print DaMuEL language coverage from the checked-in release record, offline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from scripts.damuel_inventory.inventory import (
    RELEASE_PATH,
    coverage_summary,
    load_release,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Report coverage only; never open a dataset archive or contact a host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=RELEASE_PATH)
    arguments = parser.parse_args(argv)
    try:
        release = load_release(arguments.release)
    except (OSError, UnicodeError, ValidationError) as error:
        print(f"Invalid DaMuEL release record: {error}", file=sys.stderr)
        return 2
    print(json.dumps(coverage_summary(release), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
