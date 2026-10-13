import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/g5k/prepare_gazetteer.sh"


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
    env.update(
        {
            "GEOPARSER_GAZETTEERS_DIR": str(tmp_path / scratch_name / "gazetteers"),
            "GEOPARSER_GAZETTEER_ARTIFACT_DIR": str(tmp_path / "results" / "artifacts"),
            "GAZETTEER_CALL_LOG": str(tmp_path / "install.log"),
            "PATH": os.pathsep.join((str(fake_bin), env["PATH"])),
        }
    )
    subprocess.run(["bash", str(SCRIPT)], check=True, env=env, capture_output=True)
    return Path(env["GEOPARSER_GAZETTEERS_DIR"]) / "geonames.db"


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
