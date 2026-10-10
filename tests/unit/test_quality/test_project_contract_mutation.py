"""Contract tests for the mutation workflow, the exact-mutant replay, and the DuckDB spatial setup."""

from typing import Any

import yaml

from tests.conftest import PROJECT_ROOT
from tests.unit import test_docs as docs_guard
from tests.unit.test_quality.project_contract_support import (
    job_steps,
    load_pyproject,
    named_step,
)


def test_mutation_runner_copies_quality_support_modules() -> None:
    project = load_pyproject()

    copied_paths = set(project["tool"]["mutmut"]["also_copy"])
    assert {
        "docs",
        "scripts",
        ".github",
        ".dockerignore",
        "CITATION.cff",
        "Dockerfile",
        ".pre-commit-config.yaml",
        "CHANGELOG.md",
        "mkdocs.yml",
        "MUTATION_TESTING.md",
        "benchmark-evidence/panx/feasibility-2026-09-30",
    } <= copied_paths
    # The documentation guard reads these public surfaces directly, so a
    # mutant run that left them behind would fail for want of a file rather
    # than because a mutant survived.
    assert set(docs_guard.PUBLIC_ROOTS) <= copied_paths


def test_quality_workflow_skips_mutation_only_for_scheduled_runs() -> None:
    quality = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    )
    quality_steps = quality["jobs"]["quality"]["steps"]
    quality_run = next(
        step["run"]
        for step in quality_steps
        if step.get("name") == "Run the complete deterministic quality gauntlet"
    )
    scheduled_run = quality_run.split("else", maxsplit=1)[0]
    assert "--skip-mutation" not in scheduled_run
    assert "--skip-mutation" in quality_run


def _mutation_report_text() -> str:
    return " ".join(
        (PROJECT_ROOT / "MUTATION_TESTING.md").read_text(encoding="utf-8").split()
    )


def test_mutation_report_records_the_latest_full_sweep() -> None:
    report = _mutation_report_text()
    assert "run 36901916301" in report
    assert "PR head `a532d5eba284ca5eb333cae6515513499ab02812`" in report
    assert "newest completed full mutation evidence" in report
    assert "all **3,724 mutants**" in report


def test_mutation_report_keeps_timeout_results_inconclusive() -> None:
    report = _mutation_report_text()
    assert "3,700 killed and 36 inconclusive timeouts, not 3,736 kills" in report
    assert "These six are **not counted as killed**" in report
    assert "not a claim of a new full 3,724-mutant sweep" in report


def test_mutation_report_keeps_historical_no_tests_allowance() -> None:
    report = _mutation_report_text()
    assert "`--max-no-tests 69` allowance" in report


def _changed_mutation_job() -> tuple[dict[str, Any], list[str]]:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["changed-mutation"]
    commands = [step.get("run", "") for step in job["steps"]]
    return job, commands


def _has_command(commands: list[str], fragment: str) -> bool:
    """Whether any workflow step contains the requested shell fragment."""
    return any(fragment in command for command in commands)


def test_changed_mutation_job_is_pull_request_scoped() -> None:
    job, _ = _changed_mutation_job()
    assert "github.event_name == 'pull_request'" in job["if"]


def test_changed_mutation_job_selects_modules_and_runs_mutmut() -> None:
    _, commands = _changed_mutation_job()
    assert _has_command(commands, "changed_mutation_patterns.py")
    assert _has_command(commands, "mutmut run")


def test_changed_mutation_job_gates_and_reports_mutant_results() -> None:
    _, commands = _changed_mutation_job()
    assert _has_command(commands, "--max-no-tests 0")
    assert _has_command(commands, "mutation_gate.py")
    assert _has_command(commands, "mutation_evidence.py")


def _mutation_evidence_artifact_step() -> dict[str, Any]:
    job, _ = _changed_mutation_job()
    return next(
        step
        for step in job["steps"]
        if step.get("name") == "Upload per-mutant evidence"
    )


def test_changed_mutation_job_uploads_evidence_even_on_failure() -> None:
    artifact_step = _mutation_evidence_artifact_step()

    assert artifact_step["if"] == "always()"


def test_changed_mutation_artifact_name_identifies_run_and_head() -> None:
    artifact_step = _mutation_evidence_artifact_step()

    assert (
        "mutation-evidence-${{ github.run_id }}-${{ github.event.pull_request.head.sha }}"
        in (artifact_step["with"]["name"])
    )


def test_changed_mutation_artifact_keeps_exact_run_files() -> None:
    artifact_step = _mutation_evidence_artifact_step()

    assert artifact_step["with"]["path"] == "mutation-evidence/"
    assert artifact_step["with"]["retention-days"] == "90"


def test_quality_workflow_waits_for_changed_mutation() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["quality"]

    assert job["needs"] == ["changed-mutation"]
    assert "always()" in job["if"]


def test_quality_workflow_fails_closed_when_mutation_does_not_pass() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["quality"]
    guard = job["steps"][0]
    assert "always()" in guard["if"]
    assert "needs.changed-mutation.result != 'success'" in guard["if"]
    assert "exit 1" in guard["run"]


def test_quality_workflow_dispatch_keeps_quality_as_the_default_mode() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]

    assert inputs["mode"]["default"] == "quality"
    assert "targeted-mutant-replay" in inputs["mode"]["options"]
    assert inputs["mutant_ids"]["required"] == "false"
    assert inputs["expected_sha"]["required"] == "false"


def test_exact_mutant_replay_requires_manual_dispatch() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["exact-mutant-replay"]

    assert "workflow_dispatch" in job["if"]
    assert "targeted-mutant-replay" in job["if"]


def test_exact_mutant_replay_has_a_bounded_job_and_script_step() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["exact-mutant-replay"]
    commands = [step.get("run", "") for step in job["steps"]]

    assert job["timeout-minutes"] == "240"
    assert any("scripts/mutation_replay.py" in command for command in commands)


def test_exact_mutant_replay_runs_serially_without_timeout_override() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["exact-mutant-replay"]
    commands = [step.get("run", "") for step in job["steps"]]
    replay_script = (PROJECT_ROOT / "scripts" / "mutation_replay.py").read_text(
        encoding="utf-8"
    )
    assert '_command("run", "--max-children", "1", *selected)' in replay_script
    assert all("timeout-factor" not in command for command in commands)


def test_targeted_replay_does_not_also_run_the_quality_gauntlet() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    quality_if = workflow["jobs"]["quality"]["if"]

    assert "github.event_name != 'workflow_dispatch'" in quality_if
    assert "inputs.mode != 'targeted-mutant-replay'" in quality_if


def test_exact_replay_checks_out_requested_source_without_persisting_credentials() -> (
    None
):
    steps = job_steps("quality.yml", "exact-mutant-replay")
    checkout = named_step(steps, "Check out the validated evidence source")
    assert checkout["with"]["ref"] == "${{ inputs.expected_sha }}"
    assert checkout["with"]["persist-credentials"] == "false"


def test_exact_replay_verifies_source_before_installing_dependencies() -> None:
    steps = job_steps("quality.yml", "exact-mutant-replay")
    checkout = named_step(steps, "Check out the validated evidence source")
    verify = named_step(
        steps, "Verify the actual source checkout before dependency installation"
    )
    install = named_step(steps, "Install locked test dependencies")
    assert steps.index(checkout) < steps.index(verify) < steps.index(install)
    assert "$(git rev-parse HEAD)" in verify["run"]


def test_exact_replay_uses_retained_controller_and_actual_source_identity() -> None:
    replay = named_step(
        job_steps("quality.yml", "exact-mutant-replay"),
        "Replay selected timeout mutants",
    )
    assert "$(git rev-parse HEAD)" in replay["run"]
    assert "$helper/scripts/mutation_replay.py" in replay["run"]


def _changed_mutation_script() -> str:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    return next(
        step["run"]
        for step in workflow["jobs"]["changed-mutation"]["steps"]
        if step.get("name") == "Mutate changed package modules"
    )


def test_test_only_changes_run_the_full_mutation_sweep() -> None:
    script = _changed_mutation_script()
    assert '"FULL_MUTATION"' in script
    assert "--max-no-tests 69" in script


def test_changed_mutation_job_has_a_bounded_timeout() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert workflow["jobs"]["changed-mutation"]["timeout-minutes"] == "240"


def test_changed_mutation_runs_serially_to_avoid_pytest_temp_races() -> None:
    assert _changed_mutation_script().count("mutmut run --max-children 1") == 2


def test_changed_mutation_job_installs_project_and_test_dependencies() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    install_step = next(
        step
        for step in workflow["jobs"]["changed-mutation"]["steps"]
        if step.get("name") == "Install locked test dependencies"
    )

    assert install_step["run"] == "uv sync --locked --no-default-groups --group test"


def _duckdb_spatial_steps() -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any]
]:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["changed-mutation"]["steps"]
    install_step = next(
        step for step in steps if step.get("name") == "Install DuckDB spatial extension"
    )
    mutation_step = next(
        step for step in steps if step.get("name") == "Mutate changed package modules"
    )
    return steps, install_step, mutation_step


def test_changed_mutation_job_installs_duckdb_spatial_extension() -> None:
    _, install_step, _ = _duckdb_spatial_steps()
    assert "install_extension('spatial')" in install_step["run"]


def test_spatial_extension_installs_before_mutation() -> None:
    steps, install_step, mutation_step = _duckdb_spatial_steps()
    assert steps.index(install_step) < steps.index(mutation_step)


def test_quality_gate_installs_spatial_before_its_test_suite() -> None:
    steps = job_steps("quality.yml", "quality")
    install_step = named_step(steps, "Install DuckDB spatial extension")
    quality_step = named_step(steps, "Run the complete deterministic quality gauntlet")

    assert "install_extension('spatial')" in install_step["run"]
    assert steps.index(install_step) < steps.index(quality_step)


def test_quality_workflow_skips_the_full_mutation_sweep_on_pull_requests() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    quality_step = next(
        step
        for step in workflow["jobs"]["quality"]["steps"]
        if step.get("name") == "Run the complete deterministic quality gauntlet"
    )

    assert "quality_gauntlet.py --skip-docker --skip-mutation" in quality_step["run"]
