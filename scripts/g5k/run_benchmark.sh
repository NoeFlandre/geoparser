#!/usr/bin/env bash
# The job script: runs on a reserved Grid'5000 node, not on the frontend.
#
# Submitted by submit.sh. It does three things in order:
#   1. builds the GeoNames gazetteer on the node's local disk
#   2. runs the benchmark, checkpointing into $HOME as it goes
#   3. leaves the node's local disk clean
#
# The gazetteer goes on local disk because it needs about 31 GB while building
# and 10 GB afterwards, against a 24 GB /home quota. Local disk is wiped
# between jobs, so it is rebuilt each time -- about 25 minutes, which is the
# price of not filling a shared, un-backed-up filesystem.
#
# The benchmark checkpoints per chunk into $RESULTS, which IS on /home, so a
# job that ends at its walltime is resumed rather than restarted by the next.

set -euo pipefail

VENV="${VENV:-$HOME/geoparser-venv}"
REPO_ROOT="${REPO_ROOT:-$HOME/geoparser}"
RESULTS="${RESULTS:-$HOME/geoparser-bench}"
LIMIT="${LIMIT:-}"
DEVICE="${DEVICE:-auto}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
MIN_SIMILARITY="${MIN_SIMILARITY:-0.0}"

# Node-local scratch. OAR gives each job a directory under /tmp; fall back to
# a job-named directory so two jobs on one node cannot collide.
SCRATCH="${SCRATCH:-/tmp/geoparser-${OAR_JOB_ID:-manual}}"

echo "=== job ${OAR_JOB_ID:-manual} on $(hostname) at $(date -Is) ==="
mkdir -p "$SCRATCH" "$RESULTS"

cleanup() {
    # Only ever removes this job's own scratch directory.
    echo "=== cleaning $SCRATCH at $(date -Is) ==="
    rm -rf "$SCRATCH"
    df -h /tmp | tail -1
}
trap cleanup EXIT

# Keep every large, regenerable artifact off /home.
export GEOPARSER_GAZETTEERS_DIR="$SCRATCH/gazetteers"
export XDG_DATA_HOME="$SCRATCH/share"
export XDG_CACHE_HOME="$SCRATCH/cache"
export HF_HOME="$SCRATCH/huggingface"
export GEOPARSER_DB_PATH="$SCRATCH/benchmark.sqlite"
mkdir -p "$GEOPARSER_GAZETTEERS_DIR" "$HF_HOME"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO_ROOT"

echo "=== environment ==="
python - <<'PY'
import torch
print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
df -h "$SCRATCH" | tail -1

echo "=== building gazetteer on node-local disk ==="
if [[ ! -e "$GEOPARSER_GAZETTEERS_DIR/geonames.duckdb" ]]; then
    time python -m geoparser install geonames
fi
python -m geoparser list

echo "=== running benchmark ==="
ARGS=(--output-dir "$RESULTS" --device "$DEVICE" --chunk-size "$CHUNK_SIZE"
      --min-similarity "$MIN_SIMILARITY")
if [[ -n "${PIPELINES:-}" ]]; then
    IFS=',' read -r -a selected_pipelines <<< "$PIPELINES"
    for pipeline in "${selected_pipelines[@]}"; do
        ARGS+=(--pipeline "$pipeline")
    done
fi
if [[ -n "$LIMIT" ]]; then
    ARGS+=(--limit "$LIMIT")
fi

# stdbuf so progress reaches the log while the job is still running, which is
# the only way to tell a slow run from a stuck one before the walltime ends.
stdbuf -oL -eL python -m scripts.benchmark "${ARGS[@]}"

echo "=== done at $(date -Is) ==="
ls -la "$RESULTS"
