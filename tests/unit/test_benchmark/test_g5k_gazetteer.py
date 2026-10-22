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


def _run_submit(env):
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("The Grid5000 submit regression requires Bash")
    return subprocess.run(
        [bash, str(SUBMIT_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


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
        '#!/bin/sh\nprintf \'%s\n\' "$@" > "$OARSUB_CAPTURE"\n',
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
            "RUN_OUTPUT_DIR": str(tmp_path / "results" / "evidence" / "shard-a"),
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
            "SCORING_VIEW": "newsli-exact-full-record-dedup-v1",
            "GEOPARSER_GAZETTEER_BUILD_MEMORY_LIMIT_MB": "6144",
        }
    )
    _run_submit(env)

    command = (tmp_path / "oarsub-args.txt").read_text(encoding="utf-8")
    expected = (
        "host=1/gpu=1/core=4,walltime=2:00",
        "GEOPARSER_GAZETTEER_BUILD_MEMORY_LIMIT_MB='6144'",
        f"RUN_OUTPUT_DIR='{tmp_path}/results/evidence/shard-a'",
        "SCORING_VIEW='newsli-exact-full-record-dedup-v1'",
    )
    assert all(argument in command for argument in expected)


@pytest.mark.skipif(os.name == "nt", reason="Grid5000 submission requires POSIX shell")
def test_g5k_submit_rejects_unknown_scoring_view_before_submission(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    oarstat = fake_bin / "oarstat"
    oarstat.write_text("#!/bin/sh\nprintf '[]\\n'\n", encoding="utf-8")
    oarstat.chmod(0o755)
    oarsub_called = tmp_path / "oarsub-called"
    oarsub = fake_bin / "oarsub"
    oarsub.write_text(
        f"#!/bin/sh\nprintf called > '{oarsub_called}'\n", encoding="utf-8"
    )
    oarsub.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '/usr/bin:/bin')}",
            "RESULTS": str(tmp_path / "results"),
            "SCORING_VIEW": "unknown-view",
        }
    )

    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)], capture_output=True, text=True, env=env
    )

    assert result.returncode == 2
    assert "SCORING_VIEW must be" in result.stderr
    assert not oarsub_called.exists()


@pytest.mark.skipif(os.name == "nt", reason="Grid5000 worker requires POSIX shell")
def test_g5k_worker_forwards_view_and_shard_output_dir(tmp_path):
    """A NewsLi scoring view writes checkpoints under its own shard directory."""
    project_root = Path(__file__).resolve().parents[3]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-m" && "$2" == "scripts.benchmark" ]]; then\n'
        '  printf \'%s\\0\' "$@" > "$BENCHMARK_ARGS_CAPTURE"\n'
        "  exit 0\n"
        "fi\n"
        "cat >/dev/null\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_nvidia_smi.chmod(0o755)

    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    activate = venv / "bin" / "activate"
    activate.write_text(
        f'export PATH="{fake_bin}:$PATH"\n',
        encoding="utf-8",
    )
    results = tmp_path / "results"
    artifacts = results / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "geonames.db").write_bytes(b"test gazetteer")
    run_output = (
        results
        / "evidence"
        / "matrix-2026"
        / "newsli-exact-full-record-dedup-v1"
        / "newsli-fa"
    )
    run_output.mkdir(parents=True)
    scratch = tmp_path / "scratch"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '/usr/bin:/bin')}",
            "BENCHMARK_ARGS_CAPTURE": str(tmp_path / "benchmark-args.bin"),
            "RESULTS": str(results),
            "RUN_OUTPUT_DIR": str(run_output),
            "REPO_ROOT": str(project_root),
            "VENV": str(venv),
            "SCRATCH": str(scratch),
            "PIPELINES": (
                "gliner-minilm,gliner-jina,xlm-minilm,xlm-jina,otter-minilm,otter-jina"
            ),
            "CORPORA": "newsli-fa",
            "SCORING_VIEW": "newsli-exact-full-record-dedup-v1",
            "CHUNK_SIZE": "5",
        }
    )

    result = subprocess.run(
        ["bash", str(project_root / "scripts/g5k/run_benchmark.sh")],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    captured = (tmp_path / "benchmark-args.bin").read_bytes().split(b"\0")[:-1]
    args = [part.decode() for part in captured]
    forwarded = (
        args[:2],
        args[args.index("--output-dir") + 1],
        args[args.index("--scoring-view") + 1],
        args[args.index("--corpus") + 1],
        args[args.index("--chunk-size") + 1],
    )
    assert forwarded == (
        ["-m", "scripts.benchmark"],
        str(run_output),
        "newsli-exact-full-record-dedup-v1",
        "newsli-fa",
        "5",
    )


@pytest.mark.skipif(os.name == "nt", reason="Grid5000 submission requires POSIX shell")
@pytest.mark.parametrize(
    "oarstat_script",
    [
        "exit 7\n",
        "printf '%s\\n' 'not-json'\n",
        'printf \'%s\\n\' \'[{"name":"geoparser-guard-test","state":"Running","Job_Id":71}]\'\n',
    ],
)
def test_g5k_submit_fails_closed_if_duplicate_guard_fails_or_finds_a_job(
    tmp_path, oarstat_script
):
    """A failed query, malformed response, or duplicate never reaches oarsub."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    oarstat = fake_bin / "oarstat"
    oarstat.write_text(f"#!/bin/sh\n{oarstat_script}", encoding="utf-8")
    oarstat.chmod(0o755)
    oarsub = fake_bin / "oarsub"
    oarsub.write_text('#!/bin/sh\nprintf called > "$OARSUB_CALLED"\n', encoding="utf-8")
    oarsub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '/usr/bin:/bin')}",
            "OARSUB_CALLED": str(tmp_path / "oarsub-called"),
            "JOB_NAME": "geoparser-guard-test",
            "RESULTS": str(tmp_path / "results"),
            "REPO_ROOT": str(tmp_path / "repo"),
            "VENV": str(tmp_path / "venv"),
            "CLUSTER": "grue",
            "GPUS": "1",
            "CORES": "4",
            "WALLTIME": "2:00",
        }
    )
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("The Grid5000 submit regression requires Bash")
    completed = subprocess.run(
        [bash, str(SUBMIT_SCRIPT)], capture_output=True, text=True, env=env
    )

    assert completed.returncode != 0
    assert not (tmp_path / "oarsub-called").exists()
