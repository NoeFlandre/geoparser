"""
Pin the UniTopRank code this adapter may execute, and verify a checkout of it.

UniTopRank (Hu et al., IJGIS 2026) publishes no tagged release, so the pin is
the commit of its GitLab repository that was reviewed. Each file the adapter
imports is pinned by its Git blob ID: the SHA-1 Git computes over
``"blob <size>\\0" + content``. A checkout is verified from its bytes alone,
without Git and without the network, and an unverified checkout is never
imported.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import sys
import typing as t
from pathlib import Path

REPOSITORY = "https://gitlab.com/dlr-dw/UniTopRank"
COMMIT = "346deb166f12b0c97c8c0f9759b593ff178ceafd"
LICENSE_SPDX = "Apache-2.0"

# Every file of the reviewed tree that the direct ranking import reaches, plus
# the licence and the files reviewed but not imported. The blob IDs are the
# ones GitLab's repository tree API reports for COMMIT. Every Python file in the
# tree is listed, so that none can shadow an import (see _unreviewed_modules).
REVIEWED_BLOBS: dict[str, str] = {
    "__init__.py": "7b2e6729b2d85906e7cc9bb9ae9402bb04f59a4d",
    "LICENSE": "5d817e2e7a576e60810d167d17e37d8c9e57dd20",
    "NOTICE": "102525039689d30b0e1b8f91577e25b0c473402b",
    "THIRD_PARTY_LICENSES.md": "ef488c3826b02c2465e723c87c1293a1ba6ffe52",
    "dependencies.md": "a9103631e062e90aaebfc6c55f40c7b3db6ffac0",
    "requirements.txt": "6b82771cefeaec461d181992e48865f660a12bac",
    "requirements-ner.txt": "2d13ba5012c3fa08a47a8a0e31ebd409745b933f",
    "geo_rank_api.py": "108c5e8ac8ceea53a645ae01541ac93976f5fddc",
    "geoparsing_api.py": "f323b234faa414b293c268f70737b758fdb7eb1d",
    "thread_weight_rank_algorithm_3_beam.py": (
        "d110989ed3abd2382b4af250a616e9cd9c3272ae"
    ),
    "unitorank/__init__.py": "1cf57de1dfe3c8b31913964b9c962823a9a63a77",
    "unitorank/ranker.py": "b34c490fdcd15809857d3440812d08b72be198db",
    "unitorank/types.py": "815cbcde11c55a2ec2baf9bdf2e96cc1c886c536",
    "unitorank/candidate_retriever.py": "8a7e3f2fcbc1d839885f71212e58409a6e40694f",
    "unitorank/pipeline.py": "0c3df1d34d3a1866d24bf6ab883c140f6d78c5e5",
    "unitorank/ner.py": "a5d002a8326f7719fe888565ddb5fac46a2dd444",
}


class UpstreamMismatchError(RuntimeError):
    """A checkout differs from the reviewed tree, so it must not be imported."""


# Suffixes of every file type the import system loads: source, bytecode and
# compiled extensions.
IMPORTABLE_SUFFIXES = tuple(importlib.machinery.all_suffixes())


def git_blob_id(content: bytes) -> str:
    """Return the Git blob ID of file content, as ``git hash-object`` computes it."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def verify_checkout(
    checkout: Path, blobs: t.Mapping[str, str] = REVIEWED_BLOBS
) -> list[str]:
    """
    Compare a checkout with the reviewed blob IDs.

    Args:
        checkout: Directory holding the UniTopRank tree at COMMIT
        blobs: Relative path to the expected blob ID

    Returns:
        One message per missing or changed file, and per importable file the
        blobs do not name; empty when the tree matches
    """
    problems = []
    for relative, expected in sorted(blobs.items()):
        path = checkout / relative
        if not path.is_file():
            problems.append(f"missing: {relative}")
            continue
        actual = git_blob_id(path.read_bytes())
        if actual != expected:
            problems.append(f"changed: {relative} is {actual}, pinned {expected}")
    problems.extend(_unreviewed_modules(checkout, blobs))
    problems.extend(f"symlink: {name}" for name in _symlinks(checkout))
    return problems


def _symlinks(checkout: Path) -> list[str]:
    """Return the sorted POSIX paths of symbolic links outside Git's metadata."""
    names = (
        path.relative_to(checkout).as_posix()
        for path in checkout.rglob("*")
        if path.is_symlink()
    )
    # A symlinked directory can hide importable files outside the listing above.
    return sorted(name for name in names if not name.startswith(".git/"))


def _unreviewed_modules(checkout: Path, blobs: t.Mapping[str, str]) -> list[str]:
    """
    Name each importable file in the checkout that the blob IDs do not cover.

    The checkout is placed first on the import path, so a file such as
    ``rapidfuzz.py`` would satisfy an import meant for the installed package and
    run unreviewed code.
    """
    unpinned = [name for name in _importable_names(checkout) if name not in blobs]
    return [f"unreviewed: {name}" for name in unpinned]


def _importable_names(checkout: Path) -> list[str]:
    """Return the sorted POSIX paths of importable names, outside Git's metadata."""
    names = (
        path.relative_to(checkout).as_posix()
        for path in checkout.rglob("*")
        if path.name.endswith(IMPORTABLE_SUFFIXES)
    )
    # Git's metadata directory is never on the import path.
    return sorted(name for name in names if not name.startswith(".git/"))


def load_rank_toponyms(checkout: Path) -> tuple[t.Callable[..., t.Any], t.Any]:
    """
    Verify a checkout, then import its direct ranking API.

    Only ``unitorank.ranker`` is named. ``geo_rank_api`` does not import at the
    pinned commit. The package ``__init__`` also imports the candidate retriever
    and the NER backends, but imports nothing that connects or downloads, and
    none of those objects is called here.

    Args:
        checkout: Directory holding the verified UniTopRank tree

    Returns:
        The ``rank_toponyms`` function and the ``RankerConfig`` class

    Raises:
        UpstreamMismatchError: If a module of the pinned tree is already imported
            in this interpreter, from wherever it came, if the checkout differs
            from the pin, or if ``unitorank`` resolves outside the checkout. These
            checks run before anything is imported, so no cached or foreign code
            runs.
    """
    _refuse_cached_modules()
    problems = verify_checkout(checkout)
    if problems:
        message = "; ".join(problems)
        raise UpstreamMismatchError(message)
    root = checkout.resolve()
    _place_first_on_path(root)
    origin = _unitorank_origin()
    if origin is None or not origin.is_relative_to(root):
        message = f"unitorank is imported from {origin}, not {root}"
        raise UpstreamMismatchError(message)
    ranker = _import_without_bytecode("unitorank.ranker")
    return ranker.rank_toponyms, ranker.RankerConfig


def _refuse_cached_modules() -> None:
    """
    Refuse when any module of the pinned tree is already imported.

    Python reuses a module that is already in ``sys.modules`` without reading the
    disk again. A copy from another directory would run in place of the verified
    code. So would a copy from this checkout that was imported before verification
    or before the files were updated in place. Only a fresh interpreter can load
    the verified tree, so the origin of a cached module does not matter.

    Raises:
        UpstreamMismatchError: Naming each cached module and where it came from
    """
    problems = []
    for name in provided_module_names(REVIEWED_BLOBS):
        module = sys.modules.get(name)
        if module is not None:
            origin = getattr(module, "__file__", None)
            problems.append(
                f"{name} is already imported from {origin}; "
                "import the checkout in a fresh interpreter"
            )
    if problems:
        raise UpstreamMismatchError("; ".join(problems))


def provided_module_names(blobs: t.Mapping[str, str]) -> list[str]:
    """
    Return the sorted import names of the modules the pinned tree provides.

    ``unitorank/ranker.py`` provides ``unitorank.ranker``, and a package's
    ``__init__.py`` provides the package name. The root ``__init__.py`` names no
    module, so it contributes nothing.
    """
    names: set[str] = set()
    for relative in blobs:
        if not relative.endswith(".py"):
            continue
        parts = relative.removesuffix(".py").split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            names.add(".".join(parts))
    return sorted(names)


def _place_first_on_path(directory: Path) -> None:
    """
    Put a directory first on sys.path, even when it is already on the path.

    An absolute import in the pinned code, such as thread_weight_rank_algorithm_3_beam,
    would otherwise resolve from an earlier entry. Every other entry that names the
    directory, in any spelling, is removed before it is inserted at the front.
    """
    others = [entry for entry in sys.path if Path(entry).resolve() != directory]
    sys.path[:] = [str(directory), *others]


def _import_without_bytecode(name: str) -> t.Any:
    """
    Import a module without writing its bytecode into the verified checkout.

    Bytecode written there would be an importable file the pin does not name, so
    the next verification would refuse the checkout.
    """
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module(name)
    finally:
        sys.dont_write_bytecode = previous


def _unitorank_origin() -> Path | None:
    """
    Return the file the top-level unitorank package comes from, without importing it.

    An already imported package is reported from its ``__file__``. Otherwise the
    import system only searches for the package; its ``__init__`` does not run.
    """
    loaded = sys.modules.get("unitorank")
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
    else:
        spec = importlib.util.find_spec("unitorank")
        origin = spec.origin if spec is not None else None
    return None if origin is None else Path(origin).resolve()
