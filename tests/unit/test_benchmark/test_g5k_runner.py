import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "g5k" / "run_benchmark.sh"
SUBMIT_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "g5k" / "submit.sh"


@pytest.mark.skipif(os.name == "nt", reason="needs POSIX bash; Windows bash is WSL")
def test_missing_venv_exits_2_before_side_effects(tmp_path: Path) -> None:
    venv = tmp_path / "missing-venv"
    results = tmp_path / "results"
    scratch = tmp_path / "scratch"
    env = {
        **os.environ,
        "VENV": str(venv),
        "RESULTS": str(results),
        "SCRATCH": str(scratch),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )

    activate = venv / "bin" / "activate"
    assert result.returncode == 2, result.stderr
    assert str(activate) in result.stderr, result.stderr
    assert not results.exists()
    assert not scratch.exists()


@pytest.mark.skipif(os.name == "nt", reason="needs POSIX bash; Windows bash is WSL")
def test_g5k_runner_keeps_duckdb_home_cache_on_node_scratch(tmp_path: Path) -> None:
    """The gazetteer installer must not put DuckDB extensions in the small home quota."""
    scratch = tmp_path / "node-scratch"
    venv = tmp_path / "venv"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    repo_root = tmp_path / "repo"
    helper = repo_root / "scripts" / "g5k" / "prepare_gazetteer.sh"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        (SCRIPT.parent / "prepare_gazetteer.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    helper.chmod(0o755)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text(
        f'export PATH="{fake_bin}:$PATH"\n', encoding="utf-8"
    )
    python = fake_bin / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$HOME" >> "$HOME_CAPTURE"\n'
        'if [[ "$*" == "-" ]]; then cat >/dev/null; fi\n'
        'if [[ "$*" == "-m geoparser install geonames" ]]; then '
        'printf "smoke gazetteer" > "$GEOPARSER_GAZETTEERS_DIR/geonames.db"; fi\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    stdbuf = fake_bin / "stdbuf"
    stdbuf.write_text(
        "#!/usr/bin/env bash\n"
        'while [[ "$1" != "python" ]]; do shift; done\n'
        'exec "$@"\n',
        encoding="utf-8",
    )
    stdbuf.chmod(0o755)
    date = fake_bin / "date"
    date.write_text("#!/bin/sh\nprintf '2026-10-15T00:00:00Z\\n'\n", encoding="utf-8")
    date.chmod(0o755)
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)

    home_capture = tmp_path / "python-homes.txt"
    env = {
        **os.environ,
        "VENV": str(venv),
        "REPO_ROOT": str(repo_root),
        "RESULTS": str(tmp_path / "results"),
        "SCRATCH": str(scratch),
        "HOME_CAPTURE": str(home_capture),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert home_capture.read_text(encoding="utf-8").splitlines() == [
        str(scratch / "home"),
        str(scratch / "home"),
        str(scratch / "home"),
        str(scratch / "home"),
    ]


@pytest.mark.skipif(os.name == "nt", reason="needs POSIX bash; Windows bash is WSL")
@pytest.mark.parametrize(
    "cluster",
    [
        "chicoree",
        "drac",
        "ecotaxe",
        "estats",
        "gemini",
        "grouille",
        "hydra",
        "neowise",
        "sirius",
        "vianden",
    ],
)
def test_submit_quotes_worker_command(tmp_path: Path, cluster: str) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_command = tmp_path / "worker-command"
    captured_arguments = tmp_path / "worker-arguments"
    oarstat = fake_bin / "oarstat"
    oarstat.write_text("#!/bin/sh\nprintf '[]\n'\n")
    oarstat.chmod(0o755)
    oarsub = fake_bin / "oarsub"
    oarsub.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$CAPTURE_ARGUMENTS"\n'
        "for arg do last=$arg; done\n"
        'printf "%s" "$last" > "$CAPTURE_COMMAND"\n'
    )
    oarsub.chmod(0o755)

    repo_root = tmp_path / "repo"
    runner = repo_root / "scripts" / "g5k" / "run_benchmark.sh"
    runner.parent.mkdir(parents=True)
    execution_marker = tmp_path / "runner-executed"
    runner.write_text('#!/bin/sh\n: > "$EXPECTED_EXECUTION_MARKER"\n')
    runner.chmod(0o755)

    injection_marker = tmp_path / "shell-injection-executed"
    results = f"{tmp_path}/results'; : > {injection_marker}; #"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE_COMMAND": str(captured_command),
        "CAPTURE_ARGUMENTS": str(captured_arguments),
        "CLUSTER": cluster,
        "EXPECTED_EXECUTION_MARKER": str(execution_marker),
        "REPO_ROOT": str(repo_root),
        "VENV": str(tmp_path / "venv"),
        "RESULTS": results,
    }

    submission = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    captured = captured_arguments.read_text()
    submission_checks = (
        submission.returncode == 0,
        "\n-t\nexotic\n" in f"\n{captured}\n",
    )
    assert all(submission_checks), submission_checks

    worker = subprocess.run(
        ["bash", "-c", captured_command.read_text()],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    worker_checks = (
        worker.returncode == 0,
        not injection_marker.exists(),
        execution_marker.exists(),
    )
    assert all(worker_checks), worker_checks


@pytest.mark.skipif(os.name == "nt", reason="needs POSIX bash; Windows bash is WSL")
def test_submit_accepts_empty_oar_json_when_no_jobs_exist(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    submission_marker = tmp_path / "oarsub-called"
    oarstat = fake_bin / "oarstat"
    oarstat.write_text("#!/bin/sh\nexit 0\n")
    oarstat.chmod(0o755)
    oarsub = fake_bin / "oarsub"
    oarsub.write_text(f"#!/bin/sh\n: > '{submission_marker}'\n")
    oarsub.chmod(0o755)

    repo_root = tmp_path / "repo"
    (repo_root / "scripts" / "g5k").mkdir(parents=True)
    results = tmp_path / "results"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "JOB_NAME": "empty-job-list",
        "CLUSTER": "abacus27",
        "REPO_ROOT": str(repo_root),
        "VENV": str(tmp_path / "venv"),
        "RESULTS": str(results),
    }

    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert submission_marker.exists()
