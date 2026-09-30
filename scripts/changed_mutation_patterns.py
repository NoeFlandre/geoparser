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
        pattern = _module_pattern(path, excluded_paths)
        if pattern is None:
            continue
        patterns.add(pattern)
    return sorted(patterns)


def _module_pattern(path: str, excluded_paths: list[str]) -> str | None:
    """Resolve one changed path to a mutmut glob when it has mutants."""
    source = PurePosixPath(path)
    if not _package_python_source(source):
        return None
    if _is_excluded_source(source, excluded_paths):
        return None
    source_path = Path(source)
    if not source_path.is_file() or not _has_mutants(source):
        return None
    return _module_glob(source)


def _package_python_source(source: PurePosixPath) -> bool:
    """Whether the path names a Python source module in the package."""
    return source.parts[:1] == ("geoparser",) and source.suffix == ".py"


def _is_excluded_source(source: PurePosixPath, excluded_paths: list[str]) -> bool:
    """Whether mutmut's configured exclusions cover this changed path."""
    return any(
        fnmatch.fnmatchcase(source.as_posix(), pattern) for pattern in excluded_paths
    )


def _has_mutants(source: PurePosixPath) -> bool:
    """Whether mutmut can generate any mutant for an existing source file."""
    source_text = Path(source).read_text(encoding="utf-8")
    return bool(mutate_file_contents(source.as_posix(), source_text).mutant_names)


def _module_glob(source: PurePosixPath) -> str:
    """Build the mutmut result prefix for one package module."""
    module_parts = (*source.parts[:-1], source.stem)
    return f"{'.'.join(module_parts)}.*"


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


def _requires_full_mutation(paths: list[str]) -> bool:
    """Whether test/config changes make an incremental mutation run unsafe."""
    return _mutation_config_changed(paths) or _has_non_quality_test_change(paths)


def _mutation_config_changed(paths: list[str]) -> bool:
    """Whether a repository-wide mutation setting changed."""
    return bool({"pyproject.toml", "MUTATION_TESTING.md"} & set(paths))


def _has_non_quality_test_change(paths: list[str]) -> bool:
    """Whether a test change can affect a package mutant's outcome."""
    quality_prefixes = ("tests/unit/test_quality/", "tests/unit/test_meta/")
    return any(
        path.startswith("tests/") and not path.startswith(quality_prefixes)
        for path in paths
    )


def _print_selection(paths: list[str]) -> None:
    """Print a full-run marker or the selected changed-module patterns."""
    patterns = module_patterns_for_paths(paths)
    if _requires_full_mutation(paths):
        print("FULL_MUTATION")
    else:
        print("\n".join(patterns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("head")
    args = parser.parse_args()
    _print_selection(changed_paths(args.base, args.head))


if __name__ == "__main__":
    main()
