#!/usr/bin/env bash
# Submit the benchmark to OAR, respecting the Grid'5000 usage policy.
#
#   ./scripts/g5k/submit.sh                  # default: 1 GPU, 2 hours
#   WALLTIME=4:00 LIMIT=50 ./scripts/g5k/submit.sh
#
# Policy notes this script encodes:
#   * one GPU on one host -- the workload is single-process inference, so more
#     would sit idle while still being unavailable to everyone else
#   * an explicit, modest walltime rather than the maximum
#   * a refusal to submit when a job of this name is already queued or running,
#     because duplicate and speculative reservations are what the policy
#     specifically prohibits
#
# Check conformance before and after:  usagepolicycheck -t

set -euo pipefail

JOB_NAME="${JOB_NAME:-geoparser-bench}"
WALLTIME="${WALLTIME:-2:00}"
GPUS="${GPUS:-1}"
REPO_ROOT="${REPO_ROOT:-$HOME/geoparser}"
RESULTS="${RESULTS:-$HOME/geoparser-bench}"
CLUSTER="${CLUSTER:-}"

mkdir -p "$RESULTS/logs"

existing="$(oarstat -u --json 2>/dev/null \
    | python3 -c "
import json, sys
try:
    jobs = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for job in (jobs.values() if isinstance(jobs, dict) else jobs):
    if job.get('name') == '${JOB_NAME}' and job.get('state') in {'Waiting', 'Running', 'Launching', 'toLaunch'}:
        print(job.get('Job_Id') or job.get('id'))
" || true)"

if [[ -n "$existing" ]]; then
    echo "A '${JOB_NAME}' job is already queued or running: ${existing}" >&2
    echo "Submitting another would be a duplicate reservation. Use 'oarstat -u'," >&2
    echo "or 'oardel ${existing}' if it is stale." >&2
    exit 1
fi

RESOURCES="host=1/gpu=${GPUS},walltime=${WALLTIME}"
PROPERTY=()
if [[ -n "$CLUSTER" ]]; then
    PROPERTY=(-p "cluster='${CLUSTER}'")
fi

echo "submitting: oarsub -l ${RESOURCES} ${PROPERTY[*]-}"
oarsub \
    -n "$JOB_NAME" \
    -l "$RESOURCES" \
    "${PROPERTY[@]}" \
    -O "$RESULTS/logs/%jobid%.out" \
    -E "$RESULTS/logs/%jobid%.err" \
    --checkpoint 600 \
    --signal 12 \
    "REPO_ROOT='$REPO_ROOT' RESULTS='$RESULTS' LIMIT='${LIMIT:-}' \
DEVICE='${DEVICE:-auto}' CHUNK_SIZE='${CHUNK_SIZE:-5}' \
MIN_SIMILARITY='${MIN_SIMILARITY:-0.0}' bash $REPO_ROOT/scripts/g5k/run_benchmark.sh"

echo
echo "Watch it with:   oarstat -u"
echo "Logs:            $RESULTS/logs/"
echo "Cancel with:     oardel <job id>"
