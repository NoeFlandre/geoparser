import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/g5k/prepare_gazetteer.sh"


def _bash_path(path: Path) -> str:
    """Return a path in the form expected by the Bash used on Windows."""
    if os.name != "nt":
        return str(path)
    completed = subprocess.run(
        ["cygpath", "-u", str(path)], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _run_prepare(tmp_path: Path, scratch_name: str) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$GAZETTEER_CALL_LOG"\n'
        "printf 'same geonames bytes' > \"$GEOPARSER_GAZETTEERS_DIR/geonames.db\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    installed = tmp_path / scratch_name / "gazetteers" / "geonames.db"
    env.update(
        {
            "GEOPARSER_GAZETTEERS_DIR": _bash_path(installed.parent),
            "GEOPARSER_GAZETTEER_ARTIFACT_DIR": _bash_path(
                tmp_path / "results" / "artifacts"
            ),
            "GAZETTEER_CALL_LOG": _bash_path(tmp_path / "install.log"),
            "PATH": f"{_bash_path(fake_bin)}:/usr/bin:/bin",
        }
    )
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("The gazetteer helper regression requires Bash")
    subprocess.run([bash, _bash_path(SCRIPT)], check=True, env=env, capture_output=True)
    return installed


def test_g5k_reuses_byte_identical_gazetteer_across_jobs(tmp_path):
    first_job_installed = _run_prepare(tmp_path, "job-1-scratch")
    second_job_installed = _run_prepare(tmp_path, "job-2-scratch")

    artifact = tmp_path / "results/artifacts/geonames.db"
    assert artifact.read_bytes() == b"same geonames bytes"
    assert first_job_installed.read_bytes() == artifact.read_bytes()
    assert second_job_installed.read_bytes() == artifact.read_bytes()
    assert (tmp_path / "install.log").read_text().splitlines() == [
        "-m geoparser install geonames"
    ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX execute bits are unavailable")
def test_g5k_gazetteer_helper_is_executable():
    assert SCRIPT.stat().st_mode & 0o111


SUBMIT_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/g5k/submit.sh"


@pytest.mark.skipif(os.name == "nt", reason="Grid5000 submission requires POSIX shell")
def test_g5k_submit_forwards_memory_ceiling(tmp_path):
    """A configured installer memory cap reaches the OAR job command."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    oarstat = fake_bin / "oarstat"
    oarstat.write_text("#!/bin/sh\nprintf '[]\n'\n", encoding="utf-8")
    oarstat.chmod(0o755)
    oarsub = fake_bin / "oarsub"
    oarsub.write_text(
        "#!/bin/sh\nprintf '%s\n' \"$@\" > \"$OARSUB_CAPTURE\"\n",
        encoding="utf-8",
    )
    oarsub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '/usr/bin:/bin')}",
            "OARSUB_CAPTURE": str(tmp_path / "oarsub-args.txt"),
            "JOB_NAME": "geoparser-memory-test",
            "RESULTS": str(tmp_path / "results"),
            "REPO_ROOT": str(tmp_path / "repo"),
            "VENV": str(tmp_path / "venv"),
            "CLUSTER": "grue",
            "QUEUE": "besteffort",
            "GPUS": "1",
            "CORES": "4",
            "WALLTIME": "2:00",
            "PIPELINES": "swapped,xlm-minilm,otter-cross-minilm",
            "CORPORA": "geovirus",
            "PHASES": "recognition,resolution",
            "GEOPARSER_GAZETTEER_BUILD_MEMORY_LIMIT_MB": "6144",
        }
    )
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("The Grid5000 submit regression requires Bash")
    subprocess.run(
        [bash, str(SUBMIT_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    command = (tmp_path / "oarsub-args.txt").read_text(encoding="utf-8")
    assert "host=1/gpu=1/core=4,walltime=2:00" in command
    assert "GEOPARSER_GAZETTEER_BUILD_MEMORY_LIMIT_MB='6144'" in command
