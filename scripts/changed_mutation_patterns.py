"""Select mutmut names for changed library modules."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import PurePosixPath


def module_patterns_for_paths(paths: list[str]) -> list[str]:
    """Return sorted mutmut globs for changed Python modules in the package."""
    patterns = set()
    for path in paths:
        source = PurePosixPath(path)
        if source.parts[:1] != ("geoparser",) or source.suffix != ".py":
            continue
        module_parts = (*source.parts[:-1], source.stem)
        patterns.add(f"{'.'.join(module_parts)}.*")
    return sorted(patterns)


def changed_paths(base: str, head: str) -> list[str]:
    """List added or modified package paths between two fetched commits."""
    result = subprocess.run(
        (
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMRT",
            f"{base}...{head}",
            "--",
            "geoparser",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("head")
    args = parser.parse_args()
    print("\n".join(module_patterns_for_paths(changed_paths(args.base, args.head))))


if __name__ == "__main__":
    main()
