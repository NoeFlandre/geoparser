"""Run the repository's deterministic quality stages in order."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

Command = tuple[str, ...]


@dataclass(frozen=True)
class Stage:
    """A named group of commands that share one working directory."""

    name: str
    commands: tuple[Command, ...]
    cwd: Path


@dataclass(frozen=True)
class _TrailingStageOptions:
    """Settings shared by optional and final quality stages."""

    artifact_dir: Path
    include_baseline: bool
    skip_mutation: bool
    skip_docker: bool
    offline: bool
    docker_tag: str
    demo_docker_tag: str


def _uv(*arguments: str) -> Command:
    return ("uv", "run", "--no-sync", "--offline", *arguments)


_PACKAGE_COVERAGE_COMMAND = _uv(
    "coverage",
    "report",
    "--include=geoparser/*",
    "--fail-under=100",
)


def build_stages(  # noqa: PLR0913 - keyword-only switches mirroring the CLI flags
    root: Path,
    artifact_dir: Path,
    *,
    include_baseline: bool = False,
    skip_mutation: bool = False,
    skip_docker: bool = False,
    offline: bool = False,
    docker_tag: str = "geoparser:quality-check",
) -> list[Stage]:
    """Build the ordered quality stages for a repository checkout."""
    root = root.resolve()
    artifact_dir = artifact_dir.resolve()
    coverage_report = artifact_dir / "coverage-html"
    coverage_data = artifact_dir / ".coverage"
    lock_command = (
        ("uv", "lock", "--check-exists", "--offline")
        if offline
        else ("uv", "lock", "--check")
    )
    demo_docker_tag = f"{docker_tag}-demo"

    stages = [
        Stage(
            "ruff",
            (
                _uv("ruff", "check", "."),
                _uv("ruff", "format", "--check", "."),
            ),
            root,
        ),
        Stage("ty", (_uv("ty", "check", "geoparser", "scripts", "tests"),), root),
        Stage(
            "dependencies",
            (
                lock_command,
                # Ignores are documented in pyproject.toml's [tool.deptry].
                _uv("deptry", "."),
            ),
            root,
        ),
        Stage(
            "tests",
            (
                # Coverage stays recorded for all CRAP roots; the next command
                # applies the hard line floor to the production package.
                _uv(
                    "pytest",
                    "--cov-fail-under=0",
                    f"--cov-report=html:{coverage_report}",
                ),
                _PACKAGE_COVERAGE_COMMAND,
            ),
            root,
        ),
        Stage(
            "property",
            (_uv("pytest", "tests/property", "-m", "property", "--no-cov", "-q"),),
            root,
        ),
        Stage(
            "acceptance",
            (
                _uv(
                    "pytest",
                    "tests/acceptance",
                    "-m",
                    "acceptance",
                    "--no-cov",
                    "-q",
                ),
            ),
            root,
        ),
        Stage(
            "architecture",
            (
                _uv(
                    "python",
                    "scripts/check_architecture.py",
                    "--package",
                    "geoparser",
                ),
            ),
            root,
        ),
        Stage(
            "crap",
            (
                _uv(
                    "python",
                    "scripts/crap.py",
                    "--max-crap",
                    "6",
                    "--data-file",
                    str(coverage_data),
                ),
            ),
            root,
        ),
    ]
    _append_trailing_stages(
        stages,
        root,
        _TrailingStageOptions(
            artifact_dir=artifact_dir,
            include_baseline=include_baseline,
            skip_mutation=skip_mutation,
            skip_docker=skip_docker,
            offline=offline,
            docker_tag=docker_tag,
            demo_docker_tag=demo_docker_tag,
        ),
    )
    return stages


def _append_trailing_stages(
    stages: list[Stage],
    root: Path,
    options: _TrailingStageOptions,
) -> None:
    """Add optional diagnostics and the build/documentation smoke stage."""
    if options.include_baseline:
        stages.insert(
            0,
            Stage(
                "baseline",
                (
                    _uv("pytest", "--cov-fail-under=0"),
                    _PACKAGE_COVERAGE_COMMAND,
                ),
                root,
            ),
        )
    if not options.skip_mutation:
        stages.append(_mutation_stage(root))

    build_options = ("--offline", "--no-build-isolation") if options.offline else ()
    smoke_commands: list[Command] = [
        (
            "uv",
            "build",
            *build_options,
            "--out-dir",
            str(options.artifact_dir / "dist"),
        ),
        _uv(
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(options.artifact_dir / "site"),
        ),
    ]
    if not options.skip_docker:
        smoke_commands.extend(
            _docker_smoke_commands(options.docker_tag, options.demo_docker_tag)
        )
    stages.extend(
        [
            Stage("smoke", tuple(smoke_commands), root),
            Stage("diff-review", (("git", "diff", "--check"),), root),
        ]
    )


def _mutation_stage(root: Path) -> Stage:
    """Run mutmut and enforce the measured survivor allowance."""
    return Stage(
        "mutation",
        (
            _uv("mutmut", "run"),
            _uv("mutmut", "export-cicd-stats"),
            _uv(
                "python",
                "scripts/mutation_gate.py",
                "--max-survivors",
                "0",
                "--max-no-tests",
                "69",
                "--stats",
                "mutants/mutmut-cicd-stats.json",
            ),
        ),
        root,
    )


def _docker_smoke_commands(
    docker_tag: str, demo_docker_tag: str
) -> tuple[Command, ...]:
    """Build and launch the runtime and demo images."""
    return (
        ("docker", "build", "--file", "Dockerfile", "--tag", docker_tag, "."),
        ("docker", "run", "--rm", docker_tag),
        ("docker", "build", "--file", "demo/Dockerfile", "--tag", demo_docker_tag, "."),
        ("docker", "run", "--rm", demo_docker_tag, "jupyter", "--version"),
    )


def cleanup_docker_image(root: Path, env: dict[str, str], docker_tag: str) -> None:
    """Remove the exact temporary image used by a local smoke test."""
    subprocess.run(
        ("docker", "image", "rm", docker_tag),
        cwd=root,
        env=env,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_stages(stages: list[Stage], env: dict[str, str]) -> int:
    """Run stages in order and return the first failing command's status."""
    for stage in stages:
        print(f"\n== {stage.name} ==", flush=True)
        for command in stage.commands:
            result = subprocess.run(command, cwd=stage.cwd, env=env, check=False)
            if result.returncode:
                print(
                    f"Stage {stage.name!r} failed with exit code {result.returncode}.",
                    flush=True,
                )
                return result.returncode
    print("\nQuality gauntlet passed.", flush=True)
    return 0


@contextmanager
def _mutation_output_link(root: Path, artifact_dir: Path) -> Iterator[None]:
    """Keep mutmut's fixed output directory outside the checkout when possible."""
    repository_output = root / "mutants"
    if repository_output.exists() or repository_output.is_symlink():
        yield
        return

    external_output = artifact_dir / "mutants"
    external_output.mkdir(parents=True, exist_ok=True)
    repository_output.symlink_to(external_output, target_is_directory=True)
    try:
        yield
    finally:
        repository_output.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Run all quality stages unless an explicitly diagnostic flag is used."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Run an additional coverage test pass before the quality stages (diagnostic only).",
    )
    parser.add_argument(
        "--skip-mutation",
        action="store_true",
        help="Skip mutation testing for local diagnosis; CI must not use this.",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker commands for local diagnosis; CI must not use this.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Avoid network access and check only that uv.lock exists.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="geoparser-qa-") as directory:
        artifact_dir = Path(directory)
        docker_tag = f"geoparser:qa-{artifact_dir.name}"
        environment = os.environ.copy()
        environment.update(
            {
                "GEOPARSER_QA_ARTIFACT_DIR": str(artifact_dir),
                "TMPDIR": str(artifact_dir),
                "SQLITE_TMPDIR": str(artifact_dir),
                "COVERAGE_FILE": str(artifact_dir / ".coverage"),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "RUFF_CACHE_DIR": str(artifact_dir / "ruff-cache"),
            }
        )
        stages = build_stages(
            root,
            artifact_dir,
            include_baseline=args.include_baseline,
            skip_mutation=args.skip_mutation,
            skip_docker=args.skip_docker,
            offline=args.offline,
            docker_tag=docker_tag,
        )
        try:
            with _mutation_output_link(root, artifact_dir):
                return run_stages(stages, environment)
        finally:
            if not args.skip_docker:
                cleanup_docker_image(root, environment, docker_tag)
                cleanup_docker_image(root, environment, f"{docker_tag}-demo")


if __name__ == "__main__":
    raise SystemExit(main())
