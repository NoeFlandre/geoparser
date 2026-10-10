"""Contract tests for the runtime image, the CPU-only torch index, and the lock file."""

from typing import Any

import yaml

from tests.conftest import PROJECT_ROOT
from tests.unit.test_quality.project_contract_support import load_pyproject

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI.
    import tomli as tomllib


def test_dockerfile_is_present() -> None:
    assert (PROJECT_ROOT / "Dockerfile").is_file()


def _dockerignore_patterns() -> set[str]:
    """Return the active .dockerignore patterns, without comments or blank lines."""
    lines = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_docker_ignore_excludes_local_state() -> None:
    assert (PROJECT_ROOT / ".dockerignore").is_file()
    assert {".git", ".venv", "secrets"} <= _dockerignore_patterns()


def test_pyproject_uses_the_cpu_torch_index() -> None:
    uv = load_pyproject()["tool"]["uv"]
    cpu_indexes = [index for index in uv["index"] if index["name"] == "pytorch-cpu"]
    assert [index["url"] for index in cpu_indexes] == [
        "https://download.pytorch.org/whl/cpu"
    ]
    assert any(source["index"] == "pytorch-cpu" for source in uv["sources"]["torch"])


def _project_lock() -> dict[str, Any]:
    with (PROJECT_ROOT / "uv.lock").open("rb") as lockfile:
        return tomllib.load(lockfile)


def test_lock_resolves_torch_from_the_cpu_index() -> None:
    lock = _project_lock()
    cpu_torch = [
        package
        for package in lock["package"]
        if package["name"] == "torch"
        and package.get("source", {}).get("registry")
        == "https://download.pytorch.org/whl/cpu"
    ]
    assert cpu_torch


def test_lock_omits_accelerator_packages() -> None:
    lock = _project_lock()
    assert not any(
        package["name"].startswith(("cuda-", "nvidia-")) for package in lock["package"]
    )


def test_docker_workflow_runs_the_runtime_cli() -> None:
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/docker.yml").read_text(encoding="utf-8")
    )
    commands = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "run" in step
    ]

    assert any(
        "docker run --rm geoparser:smoke --help" in command for command in commands
    )
