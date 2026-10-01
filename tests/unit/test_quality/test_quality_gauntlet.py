from pathlib import Path
from unittest.mock import Mock

from scripts.quality_gauntlet import (
    build_stages,
    cleanup_docker_image,
    main,
    run_stages,
)


def test_quality_stages_have_the_required_order(tmp_path: Path) -> None:
    stages = build_stages(Path("/repo"), tmp_path)
    assert [stage.name for stage in stages] == [
        "ruff",
        "ty",
        "dependencies",
        "tests",
        "property",
        "acceptance",
        "architecture",
        "crap",
        "mutation",
        "smoke",
        "diff-review",
    ]


def _named_stage(stages, name: str):
    """Return one quality stage by its stable name."""
    return next(stage for stage in stages if stage.name == name)


def _stage_command(stages, name: str) -> tuple[str, ...]:
    """Return the first command from one named quality stage."""
    return _named_stage(stages, name).commands[0]


def test_ty_stage_checks_all_first_party_code_roots(tmp_path: Path) -> None:
    stages = build_stages(Path("/repo"), tmp_path)
    assert _stage_command(stages, "ty")[-3:] == ("geoparser", "scripts", "tests")


def test_crap_stage_uses_the_strict_six_ceiling(tmp_path: Path) -> None:
    stages = build_stages(Path("/repo"), tmp_path)
    command = _stage_command(stages, "crap")
    assert command[command.index("--max-crap") + 1] == "6"


def test_default_quality_stages_collect_full_tree_coverage(tmp_path: Path) -> None:
    """The main and deterministic benchmark tests feed the coverage gates."""
    stages = build_stages(Path("/repo"), tmp_path)
    pytest_run, benchmark_run, package_report = _named_stage(stages, "tests").commands

    assert "pytest" in pytest_run
    assert "--cov-fail-under=0" in pytest_run
    assert benchmark_run[4:8] == (
        "pytest",
        "tests/benchmarks",
        "-m",
        "benchmark",
    )
    assert "--benchmark-disable" in benchmark_run
    assert "--cov-append" in benchmark_run
    assert "coverage" in package_report


def test_coverage_suite_uses_the_provisioned_offline_environment(
    tmp_path: Path,
) -> None:
    stages = build_stages(Path("/repo"), tmp_path)
    tests = _named_stage(stages, "tests")
    pytest_run, benchmark_run, package_report = tests.commands

    assert pytest_run[:5] == (
        "uv",
        "run",
        "--no-sync",
        "--offline",
        "pytest",
    )
    assert benchmark_run[:8] == (
        "uv",
        "run",
        "--no-sync",
        "--offline",
        "pytest",
        "tests/benchmarks",
        "-m",
        "benchmark",
    )
    assert "--benchmark-disable" in benchmark_run
    assert "--cov-append" in benchmark_run
    assert package_report == (
        "uv",
        "run",
        "--no-sync",
        "--offline",
        "coverage",
        "report",
        "--include=geoparser/*",
        "--fail-under=100",
    )


def test_diagnostic_baseline_uses_the_same_package_coverage_floor(
    tmp_path: Path,
) -> None:
    baseline = _named_stage(
        build_stages(Path("/repo"), tmp_path, include_baseline=True), "baseline"
    )

    assert "--cov-fail-under=0" in baseline.commands[0]
    assert baseline.commands[1] == (
        "uv",
        "run",
        "--no-sync",
        "--offline",
        "coverage",
        "report",
        "--include=geoparser/*",
        "--fail-under=100",
    )


def test_default_quality_stages_do_not_repeat_a_baseline_run(tmp_path: Path) -> None:
    stages = build_stages(Path("/repo"), tmp_path)

    assert "baseline" not in {stage.name for stage in stages}


def test_default_quality_stages_include_the_test_suite(tmp_path: Path) -> None:
    stages = build_stages(Path("/repo"), tmp_path)

    assert "tests" in {stage.name for stage in stages}


def test_quality_cli_can_include_an_extra_diagnostic_baseline(monkeypatch) -> None:
    """The redundant coverage pass is available only by explicit request."""
    names = []

    def fake_run_stages(stages, environment):
        names.extend(stage.name for stage in stages)
        return 0

    monkeypatch.setattr("scripts.quality_gauntlet.run_stages", fake_run_stages)

    assert main(["--include-baseline", "--skip-mutation", "--skip-docker"]) == 0
    assert names[0] == "baseline"
    assert "tests" in names


def test_dependency_stage_uses_the_documented_pyproject_config(
    tmp_path: Path,
) -> None:
    """The gauntlet and the lint job read one deptry config from pyproject.

    The allowed ignores are pinned by the project contract tests; repeating them
    on the command line would let the two lists drift apart.
    """
    stages = build_stages(Path("/repo"), tmp_path)
    dependency_stage = next(stage for stage in stages if stage.name == "dependencies")
    assert dependency_stage.commands[-1][-2:] == ("deptry", ".")


def test_uv_quality_commands_do_not_resolve_network_dependencies(
    tmp_path: Path,
) -> None:
    stages = build_stages(Path("/repo"), tmp_path)

    for stage in stages:
        for command in stage.commands:
            if command[:3] == ("uv", "run", "--no-sync"):
                assert command[3] == "--offline"


def test_offline_quality_mode_checks_for_a_lockfile_without_fetching(
    tmp_path: Path,
) -> None:
    stages = build_stages(Path("/repo"), tmp_path, offline=True)

    dependencies = next(stage for stage in stages if stage.name == "dependencies")
    assert ("uv", "lock", "--check-exists", "--offline") in dependencies.commands


def test_offline_smoke_build_reuses_the_provisioned_backend(tmp_path: Path) -> None:
    stages = build_stages(Path("/repo"), tmp_path, offline=True, skip_docker=True)

    smoke = next(stage for stage in stages if stage.name == "smoke")
    assert (
        "uv",
        "build",
        "--offline",
        "--no-build-isolation",
        "--out-dir",
        str(tmp_path / "dist"),
    ) in smoke.commands


def test_quality_stages_can_skip_expensive_local_checks(tmp_path: Path) -> None:
    stages = build_stages(Path("/repo"), tmp_path, skip_mutation=True, skip_docker=True)

    assert "mutation" not in {stage.name for stage in stages}
    smoke = next(stage for stage in stages if stage.name == "smoke")
    assert all(command[0] != "docker" for command in smoke.commands)


def test_mutation_gate_uses_the_measured_no_tests_baseline(tmp_path: Path) -> None:
    stages = build_stages(Path("/repo"), tmp_path)
    mutation_stage = next(stage for stage in stages if stage.name == "mutation")
    gate_command = mutation_stage.commands[-1]
    assert gate_command[gate_command.index("--max-no-tests") + 1] == "69"


def _smoke_stage(tmp_path: Path):
    stages = build_stages(Path("/repo"), tmp_path, docker_tag="geoparser:test")
    return next(stage for stage in stages if stage.name == "smoke")


def test_quality_runner_builds_the_requested_runtime_image(tmp_path: Path) -> None:
    smoke = _smoke_stage(tmp_path)
    assert (
        "docker",
        "build",
        "--file",
        "Dockerfile",
        "--tag",
        "geoparser:test",
        ".",
    ) in smoke.commands


def test_quality_runner_runs_the_requested_runtime_image(tmp_path: Path) -> None:
    smoke = _smoke_stage(tmp_path)
    assert ("docker", "run", "--rm", "geoparser:test") in smoke.commands


def test_quality_runner_builds_the_requested_demo_image(tmp_path: Path) -> None:
    smoke = _smoke_stage(tmp_path)
    assert (
        "docker",
        "build",
        "--file",
        "demo/Dockerfile",
        "--tag",
        "geoparser:test-demo",
        ".",
    ) in smoke.commands


def test_quality_runner_checks_the_requested_demo_image(tmp_path: Path) -> None:
    smoke = _smoke_stage(tmp_path)
    assert (
        "docker",
        "run",
        "--rm",
        "geoparser:test-demo",
        "jupyter",
        "--version",
    ) in smoke.commands


def test_quality_runner_removes_only_the_ephemeral_docker_image(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[tuple[str, ...], Path, dict[str, str], bool]] = []

    def fake_run(command, *, cwd, env, check, stdout, stderr):
        calls.append((tuple(command), cwd, env, check))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("scripts.quality_gauntlet.subprocess.run", fake_run)

    cleanup_docker_image(tmp_path, {"PATH": "test"}, "geoparser:test")

    assert calls == [
        (
            ("docker", "image", "rm", "geoparser:test"),
            tmp_path,
            {"PATH": "test"},
            False,
        )
    ]


def test_quality_runner_cleans_the_image_after_stages(monkeypatch) -> None:
    tags: list[str] = []

    monkeypatch.setattr("scripts.quality_gauntlet.run_stages", lambda stages, env: 0)
    monkeypatch.setattr(
        "scripts.quality_gauntlet.cleanup_docker_image",
        lambda root, env, docker_tag: tags.append(docker_tag),
    )

    assert main(["--skip-mutation"]) == 0
    assert len(tags) == 2
    assert tags[0].startswith("geoparser:qa-geoparser-qa-")
    assert tags[1] == f"{tags[0]}-demo"


def test_quality_runner_sets_deterministic_python_environment(monkeypatch) -> None:
    captured_environment: dict[str, str] = {}

    def fake_run_stages(stages, environment):
        captured_environment.update(environment)
        return 0

    monkeypatch.setattr("scripts.quality_gauntlet.run_stages", fake_run_stages)

    assert main(["--skip-mutation", "--skip-docker"]) == 0
    assert captured_environment["PYTHONHASHSEED"] == "0"
    assert captured_environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_quality_runner_stops_on_first_failed_command(
    monkeypatch, tmp_path: Path
) -> None:
    stages = build_stages(Path("/repo"), tmp_path)
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def fake_run(command, *, cwd, env, check):
        calls.append((tuple(command), cwd, env))
        return type("Completed", (), {"returncode": 17})()

    monkeypatch.setattr("scripts.quality_gauntlet.subprocess.run", fake_run)

    result = run_stages(stages, {"GEOPARSER_QA_ARTIFACT_DIR": str(tmp_path)})

    assert (result, calls) == (
        17,
        [
            (
                stages[0].commands[0],
                stages[0].cwd,
                {"GEOPARSER_QA_ARTIFACT_DIR": str(tmp_path)},
            )
        ],
    )


def test_coverage_failure_is_returned_without_running_later_gates(
    monkeypatch, tmp_path: Path
) -> None:
    stages = build_stages(Path("/repo"), tmp_path)
    tests = _named_stage(stages, "tests")
    property_stage = _named_stage(stages, "property")
    fail_coverage = Mock(return_value=type("Completed", (), {"returncode": 23})())
    monkeypatch.setattr("scripts.quality_gauntlet.subprocess.run", fail_coverage)

    result = run_stages([tests, property_stage], {})

    assert result == 23
    fail_coverage.assert_called_once_with(
        tests.commands[0], cwd=tests.cwd, env={}, check=False
    )


def test_package_coverage_failure_stops_before_later_gates(
    monkeypatch, tmp_path: Path
) -> None:
    stages = build_stages(Path("/repo"), tmp_path)
    tests = _named_stage(stages, "tests")
    property_stage = _named_stage(stages, "property")
    fail_package_report = Mock(
        side_effect=[
            type("Completed", (), {"returncode": 0})(),
            type("Completed", (), {"returncode": 0})(),
            type("Completed", (), {"returncode": 23})(),
        ]
    )
    monkeypatch.setattr("scripts.quality_gauntlet.subprocess.run", fail_package_report)

    result = run_stages([tests, property_stage], {})

    assert result == 23
    assert [call.args[0] for call in fail_package_report.call_args_list] == list(
        tests.commands
    )


def test_quality_runner_preserves_stage_order_and_artifact_environment(
    monkeypatch, tmp_path: Path
) -> None:
    stages = build_stages(Path("/repo"), tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_run(command, *, cwd, env, check):
        calls.append((command[0], env["GEOPARSER_QA_ARTIFACT_DIR"]))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("scripts.quality_gauntlet.subprocess.run", fake_run)

    result = run_stages(stages, {"GEOPARSER_QA_ARTIFACT_DIR": str(tmp_path)})

    assert result == 0
    assert len(calls) == sum(len(stage.commands) for stage in stages)
    assert all(artifact == str(tmp_path) for _, artifact in calls)
