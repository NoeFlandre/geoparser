#!/usr/bin/env bash
# Prepare the Python environment for the benchmark on a Grid'5000 frontend.
#
# Run once per site, from the repository root on the site frontend. The
# environment lives in $HOME so later jobs reuse it; nothing here needs a
# reserved node, so it must not be run inside a job.
#
#   ./scripts/g5k/setup.sh
#
# /home is quota-limited (24 GB soft), so this installs the CPU+CUDA wheels
# only, and the gazetteer is deliberately NOT built here -- it needs ~31 GB
# while building and belongs on a node's local disk. See run_benchmark.sh.

set -euo pipefail

VENV="${VENV:-$HOME/geoparser-venv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "repository: $REPO_ROOT"
echo "virtualenv: $VENV"

if [[ ! -d "$VENV" ]]; then
    python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --quiet --upgrade pip
# Installed from the pinned lockfile inputs in pyproject.toml. The CUDA build
# of torch is the default on PyPI for Linux, which is what the GPU nodes need;
# the project's own uv index pin selects the CPU build, so it is bypassed here.
python -m pip install --quiet -e "$REPO_ROOT"
python -m pip install --quiet \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz"

# Importing the stack here is a convenience, not a gate. Site frontends can be
# virtual machines whose CPU advertises no SSE4.2/AVX, and NumPy's wheels refuse
# to load below their x86-64-v2 baseline -- flille is one such frontend. That
# says nothing about the compute nodes, which are real hardware, and the policy
# sends this work to a node regardless. So report and carry on.
if python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda build {torch.version.cuda}")
PY
then
    echo "Frontend can import the stack."
else
    echo
    echo "NOTE: the stack does not import on this frontend. If the error"
    echo "mentions a NumPy baseline (x86-64-v2), it is the frontend's virtual"
    echo "CPU, not a broken install -- verify on a reserved node instead:"
    echo "  oarsub -I -l host=1/gpu=1,walltime=0:15"
fi

echo "Setup complete. Next: submit a job with scripts/g5k/submit.sh"
