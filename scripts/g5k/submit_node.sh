#!/usr/bin/env bash
# Submit one run_node.sh job to OAR on a Grid'5000 GPU node.
#
#   JOB_NAME=geoparser-gate-old BUNDLE=... COMMIT=... RUN_NAME=gate-old \
#   CORPORA=geovirus PIPELINES=hybrid PHASES=recognition,resolution \
#   ./scripts/g5k/submit_node.sh
#
# Policy notes, as in submit.sh:
#   * one GPU on one host, and a bounded walltime
#   * a refusal to submit when a job of the same name is already queued or
#     running, because duplicate reservations are what the policy prohibits
#   * besteffort by default: the job is interrupted when the GPU is claimed,
#     and idempotent resubmits it, resuming from the checkpoints run_node.sh
#     persisted after each corpus
#
# Check conformance before and after:  usagepolicycheck -t

set -euo pipefail

: "${JOB_NAME:?set JOB_NAME}"
: "${BUNDLE:?set BUNDLE}"
: "${COMMIT:?set COMMIT}"
: "${RUN_NAME:?set RUN_NAME}"
: "${CORPORA:?set CORPORA}"
: "${PIPELINES:?set PIPELINES}"
WALLTIME="${WALLTIME:-3:00}"
CLUSTER="${CLUSTER:-grue}"
QUEUE="${QUEUE:-besteffort}"
RESULTS="${RESULTS:-$HOME/geoparser-g5k/runs/$RUN_NAME}"
HELPERS="${HELPERS:-$HOME/geoparser-g5k/helpers}"
PHASES="${PHASES:-}"
VERIFY="${VERIFY:-1}"
DEVICE="${DEVICE:-cuda}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
MIN_SIMILARITY="${MIN_SIMILARITY:-0.0}"

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "COMMIT must be a full 40-character hash" >&2
    exit 2
fi
if [[ ! "$CORPORA" =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]]; then
    echo "CORPORA must be a comma-separated list of corpus names" >&2
    exit 2
fi
if [[ ! "$PIPELINES" =~ ^[a-z0-9.-]+(,[a-z0-9.-]+)*$ ]]; then
    echo "PIPELINES must be a comma-separated list of pipeline names" >&2
    exit 2
fi
if [[ -n "$PHASES" && ! "$PHASES" =~ ^(recognition|resolution)(,(recognition|resolution))?$ ]]; then
    echo "PHASES must be recognition, resolution, or both" >&2
    exit 2
fi

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
    echo "Use 'oarstat -u', or 'oardel ${existing}' if it is stale." >&2
    exit 1
fi

# Everything the job needs is passed in its command line, quoted for the shell.
job_env=$(printf "BUNDLE=%q COMMIT=%q RUN_NAME=%q RESULTS=%q CORPORA=%q PIPELINES=%q PHASES=%q VERIFY=%q DEVICE=%q CHUNK_SIZE=%q MIN_SIMILARITY=%q HELPERS=%q" \
    "$BUNDLE" "$COMMIT" "$RUN_NAME" "$RESULTS" "$CORPORA" "$PIPELINES" "$PHASES" \
    "$VERIFY" "$DEVICE" "$CHUNK_SIZE" "$MIN_SIMILARITY" "$HELPERS")

OPTIONS=(-p "cluster='${CLUSTER}'")
CHECKPOINT_SECONDS=600
if [[ "$QUEUE" == "besteffort" ]]; then
    OPTIONS+=(-q besteffort -t besteffort -t idempotent)
    CHECKPOINT_SECONDS=60
fi

echo "submitting: oarsub -n ${JOB_NAME} -l host=1/gpu=1,walltime=${WALLTIME} ${OPTIONS[*]}"
oarsub \
    -n "$JOB_NAME" \
    -l "host=1/gpu=1,walltime=${WALLTIME}" \
    "${OPTIONS[@]}" \
    -O "$RESULTS/logs/%jobid%.out" \
    -E "$RESULTS/logs/%jobid%.err" \
    --checkpoint "$CHECKPOINT_SECONDS" \
    --signal 12 \
    "${job_env} bash ${HELPERS}/run_node.sh"

echo
echo "Watch it with:   oarstat -u"
echo "Logs:            $RESULTS/logs/"
echo "Cancel with:     oardel <job id>"
