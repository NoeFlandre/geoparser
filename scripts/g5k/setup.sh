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

python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
PY

echo "Setup complete. Next: submit a job with scripts/g5k/submit.sh"
