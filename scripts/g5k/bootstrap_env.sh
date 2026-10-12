#!/usr/bin/env bash
# Build a Python environment for one checked-out tree, from that tree's uv.lock.
#
#   bootstrap_env.sh TREE VENV
#
# Run on a reserved node, never on a frontend, and never inside the home
# directory: the venv is node-local scratch and is rebuilt by each job.
#
# The lock selects the PyTorch CPU index on Linux (pyproject.toml, tool.uv.sources),
# which would leave a GPU node with a CPU-only torch. So the locked versions are
# installed for everything else, and torch is installed from PyPI at the locked
# version: the CUDA build the 2026-09-23 runs used.

set -euo pipefail

TREE="$1"
VENV="$2"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip

cd "$TREE"
# A tree may pin the uv it must be used with (tool.uv.required-version), and uv
# refuses to run any other version. Install exactly that one.
UV_SPEC="$("$VENV/bin/python" -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb")).get("tool", {}).get("uv", {}).get("required-version", ""))')"
"$VENV/bin/python" -m pip install --quiet "uv${UV_SPEC}"
"$VENV/bin/uv" export --frozen --no-dev --no-hashes --no-emit-project --no-header \
    > "$VENV/locked.txt"

TORCH_VERSION="$(sed -n -E 's/^torch==([^+[:space:];]+).*/\1/p' "$VENV/locked.txt" | head -1)"
if [[ -z "$TORCH_VERSION" ]]; then
    echo "bootstrap_env: no torch pin in $TREE/uv.lock" >&2
    exit 2
fi
grep -v -E '^torch([=@;[:space:]]|$)' "$VENV/locked.txt" > "$VENV/requirements.txt"

"$VENV/bin/python" -m pip install --quiet -r "$VENV/requirements.txt"
"$VENV/bin/python" -m pip install --quiet "torch==${TORCH_VERSION}"

echo "tree: $TREE"
echo "torch pin: ${TORCH_VERSION} (CUDA build from PyPI)"
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
for name in ("torch", "transformers", "huggingface-hub", "tokenizers",
             "sentence-transformers", "gliner2", "spacy", "numpy"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}: not installed")
import torch
print(f"torch.version.cuda={torch.version.cuda}")
PY
