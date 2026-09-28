"""Select mutmut names for changed library modules."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
from pathlib import Path, PurePosixPath

import toml
from mutmut.mutation.file_mutation import mutate_file_contents


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
        source_path = Path(source)
        if not source_path.is_file():
            continue
        source_text = source_path.read_text(encoding="utf-8")
        if not mutate_file_contents(source.as_posix(), source_text).mutant_names:
            continue
        module_parts = (*source.parts[:-1], source.stem)
        patterns.add(f"{'.'.join(module_parts)}.*")
    return sorted(patterns)


def changed_paths(base: str, head: str) -> list[str]:
    """List added, modified, and deleted paths between fetched commits."""
    result = subprocess.run(
        (
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMRTD",
            f"{base}...{head}",
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
    paths = changed_paths(args.base, args.head)
    patterns = module_patterns_for_paths(paths)
    test_paths = [path for path in paths if path.startswith("tests/")]
    quality_only_tests = all(
        path.startswith(("tests/unit/test_quality/", "tests/unit/test_meta/"))
        for path in test_paths
    )
    mutation_config_changed = any(
        path in {"pyproject.toml", "MUTATION_TESTING.md"} for path in paths
    )

    if mutation_config_changed or (
        test_paths and not patterns and not quality_only_tests
    ):
        print("FULL_MUTATION")
    else:
        print("\n".join(patterns))


if __name__ == "__main__":
    main()
