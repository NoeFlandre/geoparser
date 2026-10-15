import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "g5k" / "run_benchmark.sh"


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
