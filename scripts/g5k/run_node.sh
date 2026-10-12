#!/usr/bin/env bash
# Job script for one reserved Grid'5000 GPU node. Submitted by submit_node.sh.
#
# It does, in order:
#   1. clones one commit from a git bundle into node-local scratch
#   2. builds a venv from that commit's own uv.lock (bootstrap_env.sh)
#   3. builds the GeoNames gazetteer on node-local disk (about 25 minutes)
#   4. verifies the corpora it is about to score (verify_corpora.py), and stops
#      on the first mismatch
#   5. runs the benchmark one corpus at a time, persisting checkpoints and
#      reports to $RESULTS after each corpus, so an idempotent resubmission
#      resumes where the preempted job stopped
#   6. aggregates the per-corpus reports into summary.md and summary.json
#   7. removes its scratch directory, whatever happened
#
# Required: BUNDLE, COMMIT, RUN_NAME, RESULTS, CORPORA, PIPELINES.
# Optional: PHASES (recognition, resolution, or both), VERIFY (1), DEVICE (cuda),
#           CHUNK_SIZE (5), MIN_SIMILARITY (0.0), HELPERS (this directory).

set -euo pipefail

: "${BUNDLE:?set BUNDLE to a git bundle path}"
: "${COMMIT:?set COMMIT to the full commit to run}"
: "${RUN_NAME:?set RUN_NAME}"
: "${RESULTS:?set RESULTS to a directory on the frontend home}"
: "${CORPORA:?set CORPORA to comma-separated corpus names}"
: "${PIPELINES:?set PIPELINES to comma-separated pipeline names}"

HELPERS="${HELPERS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PHASES="${PHASES:-}"
VERIFY="${VERIFY:-1}"
DEVICE="${DEVICE:-cuda}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
MIN_SIMILARITY="${MIN_SIMILARITY:-0.0}"

JOB="${OAR_JOB_ID:-manual}"
SCRATCH="/tmp/geoparser-${RUN_NAME}-${JOB}"
OUT="$SCRATCH/out"
TREE="$SCRATCH/src"
VENV="$SCRATCH/venv"

mkdir -p "$SCRATCH" "$RESULTS/out" "$RESULTS/logs"

# Everything large is node-local. Nothing here touches the home quota.
export XDG_CACHE_HOME="$SCRATCH/cache"
export XDG_DATA_HOME="$SCRATCH/share"
export HF_HOME="$SCRATCH/huggingface"
export UV_CACHE_DIR="$SCRATCH/uv"
export PIP_CACHE_DIR="$SCRATCH/pip"
export TMPDIR="$SCRATCH/tmp"
export GEOPARSER_GAZETTEERS_DIR="$SCRATCH/gazetteers"
export GEOPARSER_DB_PATH="$SCRATCH/benchmark.sqlite"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$UV_CACHE_DIR" \
    "$PIP_CACHE_DIR" "$TMPDIR" "$GEOPARSER_GAZETTEERS_DIR" "$OUT"

persist() {
    # Checkpoints, reports and summaries only. Corpus caches are re-downloaded
    # and re-verified by every job, so they never go back to the frontend.
    rsync -a --prune-empty-dirs \
        --include='*/' --include='*.json' --include='*.md' --include='*.log' \
        --exclude='*' "$OUT/" "$RESULTS/out/"
}

cleanup() {
    local status=$?
    # Scratch is always removed: the gazetteer alone is about 31 GB, and the
    # node may be handed to another job. The OAR logs under $RESULTS/logs keep
    # the evidence of what failed.
    persist || echo "persist failed; checkpoints of this job may be lost" >&2
    rm -rf "$SCRATCH"
    echo "=== finished at $(date -Is) with status $status ==="
    exit "$status"
}
trap cleanup EXIT

echo "=== job $JOB on $(hostname) at $(date -Is), run $RUN_NAME ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
df -h /tmp | tail -1

# Checkpoints from an earlier, preempted job of this run come back first.
if [[ -d "$RESULTS/out" ]]; then
    rsync -a "$RESULTS/out/" "$OUT/"
fi

git clone -q "$BUNDLE" "$TREE"
git -C "$TREE" checkout -q --detach "$COMMIT"
if [[ "$(git -C "$TREE" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "checked-out commit does not match $COMMIT" >&2
    exit 2
fi
echo "commit: $COMMIT"

bash "$HELPERS/bootstrap_env.sh" "$TREE" "$VENV" 2>&1 | tee "$RESULTS/logs/${JOB}-env.log"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$TREE"

echo "=== gazetteer source headers at $(date -Is) ==="
curl -sI https://download.geonames.org/export/dump/allCountries.zip \
    | grep -i -E '^(last-modified|content-length|etag):' || true

echo "=== building gazetteer geonames on node-local disk ==="
time python -m geoparser install geonames
python -m geoparser list

CORPUS_ARGS=()
for corpus_name in ${CORPORA//,/ }; do
    CORPUS_ARGS+=("$corpus_name")
done
PIPELINE_ARGS=()
for pipeline in ${PIPELINES//,/ }; do
    PIPELINE_ARGS+=(--pipeline "$pipeline")
done
PHASE_ARGS=()
for phase in ${PHASES//,/ }; do
    PHASE_ARGS+=(--phase "$phase")
done

if [[ "$VERIFY" == "1" ]]; then
    echo "=== verifying corpora at $(date -Is) ==="
    VERIFY_ARGS=()
    for corpus_name in "${CORPUS_ARGS[@]}"; do
        VERIFY_ARGS+=(--corpus "$corpus_name")
    done
    PYTHONPATH="$TREE" python "$HELPERS/verify_corpora.py" \
        --cache "$OUT" --output "$RESULTS/logs/${JOB}-corpus-verification.json" \
        "${VERIFY_ARGS[@]}"
    persist
fi

echo "=== benchmark at $(date -Is) ==="
for corpus_name in "${CORPUS_ARGS[@]}"; do
    echo "--- $corpus_name at $(date -Is)"
    python -m scripts.benchmark \
        --output-dir "$OUT" \
        --corpus "$corpus_name" \
        "${PIPELINE_ARGS[@]}" \
        "${PHASE_ARGS[@]}" \
        --device "$DEVICE" \
        --chunk-size "$CHUNK_SIZE" \
        --min-similarity "$MIN_SIMILARITY"
    persist
done

PYTHONPATH="$TREE" python "$HELPERS/summarize.py" --root "$OUT"
persist

echo "=== model revisions resolved on the node ==="
for ref in "$HF_HOME"/hub/models--*/refs/main; do
    [[ -e "$ref" ]] || continue
    printf '%s %s\n' "$(basename "$(dirname "$(dirname "$ref")")")" "$(cat "$ref")"
done | tee "$RESULTS/logs/${JOB}-hf-revisions.txt"
