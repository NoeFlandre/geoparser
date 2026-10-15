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
def test_submit_quotes_worker_command(tmp_path: Path) -> None:
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
        "CLUSTER": "drac",
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
    assert submission.returncode == 0, submission.stderr
    arguments = captured_arguments.read_text().splitlines()
    assert any(
        left == "-t" and right == "exotic"
        for left, right in zip(arguments, arguments[1:])
    )

    worker = subprocess.run(
        ["bash", "-c", captured_command.read_text()],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert worker.returncode == 0, worker.stderr
    assert not injection_marker.exists()
    assert execution_marker.exists()
