"""
Run the pinned UniTopRank ranker in an environment built from its reviewed requirements.

The locked project environment pins a newer RapidFuzz than UniTopRank was
reviewed with, and RapidFuzz's Levenshtein scores decide the ranking. So the
ranker does not run in the project environment. It runs in a separate
interpreter, created from ``ranker-requirements.txt``, which is byte-identical to
the upstream ``requirements.txt`` that the pin verifies. Before it imports
anything from the checkout, that interpreter checks each installed version
against the reviewed one.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import typing as t
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.unitoprank_benchmark.pins import load_rank_toponyms

REQUIREMENTS = Path(__file__).with_name("ranker-requirements.txt")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Isolated mode (-I) ignores PYTHON* variables such as PYTHONPATH, and keeps the
# working directory and the user site off sys.path, so nothing outside the
# reviewed environment can shadow a package or run from site customisation. The
# repository root is then added explicitly, as the first entry.
CHILD_BOOTSTRAP = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from scripts.unitoprank_benchmark.isolated_ranker import main; "
    "sys.exit(main(sys.argv[2:]))"
)


def reviewed_versions(text: str) -> dict[str, str]:
    """
    Return the version each exact pin in a requirements file names.

    Raises:
        ValueError: If a requirement is not pinned with ``==``
    """
    pins: dict[str, str] = {}
    for requirement in text.split():
        name, separator, version = requirement.partition("==")
        if not separator:
            message = f"not an exact pin: {requirement!r}"
            raise ValueError(message)
        pins[name] = version
    return pins


def installed_mismatches(pins: Mapping[str, str]) -> list[str]:
    """Name each pinned package whose installed version differs from its pin."""
    problems = []
    for name, version in sorted(pins.items()):
        installed = importlib.metadata.version(name)
        if installed != version:
            problems.append(f"{name} is {installed}, reviewed {version}")
    return problems


def _interpreter_command(python: str | Path) -> str:
    """
    Return the interpreter as the child is launched.

    A path such as ``.venv/bin/python`` is made absolute here, from the caller's
    directory, because the child runs from the repository root. It is not resolved
    through symbolic links: a virtual environment's interpreter is a link whose
    location selects that environment. A bare command name is returned unchanged
    for PATH lookup.
    """
    text = str(python)
    # A bare name has no directory part; "./python" has one, so it is a path.
    if Path(text).name != text:
        return str(Path(text).absolute())
    return text


def rank_in_reviewed_environment(
    checkout: Path,
    python: str | Path,
    *,
    text: str,
    toponyms: Sequence[Mapping[str, t.Any]],
    candidates_by_toponym: Mapping[str, t.Any],
    config: Mapping[str, t.Any],
) -> dict[str, t.Any]:
    """
    Rank one document with the pinned ranker, run by ``python``.

    The keyword arguments match the ranking function that
    :func:`scripts.unitoprank_benchmark.ranking.rank_document` calls, except that
    ``config`` holds the ``RankerConfig`` fields rather than the object itself,
    because the configuration is built inside the reviewed environment.

    Args:
        checkout: Directory holding the verified UniTopRank tree
        python: Interpreter of the environment built from ``ranker-requirements.txt``.
            A path is read from the caller's working directory, not the repository
            root the child runs from; a bare command name is looked up on PATH.

    Returns:
        The ranked candidates by normalized toponym, under
        ``ranked_candidates_by_toponym``. That is the only part of the ranker's
        result the caller reads; the rest echoes the configuration, which can
        hold sets that JSON cannot encode.

    Raises:
        RuntimeError: If the interpreter refuses to run, for example because an
            installed version differs from the reviewed one
    """
    request = {
        "text": text,
        "toponyms": list(toponyms),
        "candidates_by_toponym": candidates_by_toponym,
        "config": dict(config),
    }
    completed = subprocess.run(
        [
            _interpreter_command(python),
            "-I",
            "-c",
            CHILD_BOOTSTRAP,
            str(REPOSITORY_ROOT),
            # The child runs from the repository root, so resolve the path here.
            str(checkout.resolve()),
        ],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        message = f"the reviewed ranker did not run: {completed.stderr.strip()}"
        raise RuntimeError(message)
    return json.loads(completed.stdout)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Read a ranking request from stdin and print the ranker's result as JSON.

    Exits with status 2, and prints nothing to stdout, when an installed version
    differs from the reviewed one, so the checkout is never imported.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, metavar="DIR")
    arguments = parser.parse_args(argv)
    problems = installed_mismatches(reviewed_versions(REQUIREMENTS.read_text()))
    if problems:
        print("; ".join(problems), file=sys.stderr)
        return 2
    request = json.load(sys.stdin)
    rank_toponyms, ranker_config = load_rank_toponyms(arguments.checkout)
    result = rank_toponyms(
        text=request["text"],
        toponyms=request["toponyms"],
        candidates_by_toponym=request["candidates_by_toponym"],
        config=ranker_config(**request["config"]),
    )
    ranking = {"ranked_candidates_by_toponym": result["ranked_candidates_by_toponym"]}
    print(json.dumps(ranking, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
