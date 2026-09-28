"""Select mutmut names for changed library modules."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
from pathlib import Path, PurePosixPath

import toml


def _excluded_paths() -> list[str]:
    """Read mutmut's excluded source paths from the repository config."""
    for parent in Path(__file__).resolve().parents:
        config_path = parent / "pyproject.toml"
        if config_path.is_file():
            config = toml.loads(config_path.read_text(encoding="utf-8"))
            return config.get("tool", {}).get("mutmut", {}).get("do_not_mutate", [])
    message = "Could not find pyproject.toml above changed_mutation_patterns.py"
    raise FileNotFoundError(message)


def module_patterns_for_paths(paths: list[str]) -> list[str]:
    """Return sorted mutmut globs for changed Python modules in the package."""
    patterns = set()
    excluded_paths = _excluded_paths()
    for path in paths:
        source = PurePosixPath(path)
        if source.parts[:1] != ("geoparser",) or source.suffix != ".py":
            continue
        if any(
            fnmatch.fnmatchcase(source.as_posix(), pattern)
            for pattern in excluded_paths
        ):
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
