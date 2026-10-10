"""Contract tests for the CI gates: triggers, required checks, concurrency, and shell and action hygiene."""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.conftest import PROJECT_ROOT
from tests.unit.test_quality.project_contract_support import UniqueKeyLoader


def _bash_executable() -> str:
    if sys.platform == "win32":
        git = shutil.which("git")
        if git is not None:
            git_bash = Path(git).parent.parent / "bin" / "bash.exe"
            if git_bash.is_file():
                return str(git_bash)
        raise FileNotFoundError("Git Bash is required to parse workflow shell scripts")
    return shutil.which("bash") or "bash"


def test_quality_workflow_runs_on_pull_requests() -> None:
    quality = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    )
    quality_triggers = quality.get("on", quality.get(True, {}))
    quality_steps = quality["jobs"]["quality"]["steps"]
    quality_commands = [step.get("run", "") for step in quality_steps]

    assert "pull_request" in quality_triggers
    assert any("scripts/quality_gauntlet.py" in command for command in quality_commands)


def test_pull_request_validation_includes_base_edits() -> None:
    """Every base edit runs genuine validation with stable required names."""
    for workflow in _pull_request_workflows():
        events = set(workflow["on"]["pull_request"]["types"])
        assert {"opened", "synchronize", "reopened", "ready_for_review"} <= events
        assert "edited" in events
        assert "workflow_dispatch" in workflow["on"]


def _pull_request_workflows() -> list[dict[str, Any]]:
    """Load workflows that receive pull-request events."""
    workflows_dir = PROJECT_ROOT / ".github/workflows"
    found = []
    for path in sorted(workflows_dir.glob("*.yml")):
        workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        if "pull_request" in workflow.get("on", {}):
            found.append(workflow)
    return found


@pytest.mark.parametrize(
    ("filename", "job", "expected"),
    [
        ("test.yml", "tests-passed", "tests-passed"),
        ("lint.yml", "ruff", "ruff"),
        ("docs.yml", "build", "build"),
        ("quality.yml", "quality", "quality-gate"),
    ],
)
def test_required_check_names_are_stable(
    filename: str, job: str, expected: str
) -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows" / filename).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert workflow["jobs"][job]["name"] == expected


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_test_gate_rejects_unsuccessful_dependencies(result: str) -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    gate = workflow["jobs"]["tests-passed"]
    assert "always()" in gate["if"]
    assert set(gate["needs"]) == {"pytest", "coverage"}
    failure_step = gate["steps"][0]
    assert failure_step["run"] == "exit 1"
    assert f"contains(needs.*.result, '{result}')" in failure_step["if"]


def test_github_workflows_have_no_duplicate_yaml_keys() -> None:
    workflows_dir = PROJECT_ROOT / ".github/workflows"
    for workflow in workflows_dir.glob("*.yml"):
        yaml.load(workflow.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)


def _is_bash_step(step: dict[str, Any]) -> bool:
    """Whether a workflow step declares an inline Bash command."""
    return step.get("shell", "bash") == "bash" and "run" in step


def _bash_script_cases() -> list[tuple[str, str, str]]:
    """Return each inline Bash command with its workflow and job labels."""
    cases = []
    workflows_dir = PROJECT_ROOT / ".github/workflows"
    for workflow_path in workflows_dir.glob("*.yml"):
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job_name, job in workflow.get("jobs", {}).items():
            cases.extend(
                (workflow_path.name, job_name, step["run"])
                for step in job["steps"]
                if _is_bash_step(step)
            )
    return cases


def _assert_bash_command_parses(
    workflow_name: str, job_name: str, command: str
) -> None:
    """Ask Bash to parse one inline command without executing it."""
    result = subprocess.run(
        [_bash_executable(), "-n"],
        input=command,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, f"{workflow_name}:{job_name}: {result.stderr}"


def test_github_workflow_bash_scripts_parse() -> None:
    for workflow_name, job_name, command in _bash_script_cases():
        _assert_bash_command_parses(
            workflow_name,
            job_name,
            command,
        )


def test_windows_uses_git_bash_instead_of_the_wsl_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = tmp_path / "Git" / "cmd" / "git.exe"
    git.parent.mkdir(parents=True)
    git.touch()
    git_bash = tmp_path / "Git" / "bin" / "bash.exe"
    git_bash.parent.mkdir(parents=True)
    git_bash.touch()

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        shutil,
        "which",
        lambda command: (
            str(git) if command == "git" else r"C:\Windows\System32\bash.exe"
        ),
    )

    assert _bash_executable() == str(git_bash)


def _action_ref_is_unpinned(ref: str) -> bool:
    """Whether an external action is not pinned to a full commit SHA."""
    return not ref.startswith("./") and not re.search(r"@[0-9a-f]{40}$", ref)


def _unpinned_workflow_actions() -> list[str]:
    """Collect non-local workflow actions without immutable revisions."""
    uses = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)", re.MULTILINE)
    unpinned = []
    for path in sorted((PROJECT_ROOT / ".github/workflows").glob("*.yml")):
        unpinned.extend(
            f"{path.name}: {ref}"
            for ref in uses.findall(path.read_text(encoding="utf-8"))
            if _action_ref_is_unpinned(ref)
        )
    return unpinned


def test_every_action_is_pinned_to_a_commit_sha() -> None:
    assert _unpinned_workflow_actions() == []


@pytest.mark.parametrize(
    "filename",
    [
        "benchmark.yml",
        "docker.yml",
        "lint.yml",
        "quality.yml",
        "security.yml",
        "test.yml",
    ],
)
def test_workflow_concurrency_separates_push_from_pull_request(filename: str) -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows" / filename).read_text(),
        Loader=yaml.BaseLoader,
    )
    assert "github.event_name" in workflow["concurrency"]["group"]
    assert (
        "github.event.pull_request.state == 'open'"
        in workflow["concurrency"]["cancel-in-progress"]
    )


def test_documentation_build_concurrency_separates_event_types() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/docs.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    concurrency = workflow["jobs"]["build"]["concurrency"]
    assert "github.event_name" in concurrency["group"]
    assert (
        "github.event.pull_request.state == 'open'" in concurrency["cancel-in-progress"]
    )


@pytest.mark.parametrize(
    ("filename", "job"),
    [
        ("benchmark.yml", "compare"),
        ("docker.yml", "docker-smoke"),
        ("docs.yml", "build"),
        ("lint.yml", "ruff"),
        ("lint.yml", "ty"),
        ("quality.yml", "quality"),
        ("quality.yml", "changed-mutation"),
        ("security.yml", "dependency-audit"),
        ("security.yml", "codeql"),
        ("security.yml", "workflow-lint"),
        ("test.yml", "pytest"),
        ("test.yml", "coverage"),
        ("test.yml", "tests-passed"),
    ],
)
def test_closed_pr_edits_do_not_repeat_validation(filename: str, job: str) -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows" / filename).read_text(),
        Loader=yaml.BaseLoader,
    )
    assert "github.event.pull_request.state == 'open'" in workflow["jobs"][job]["if"]
