#!/usr/bin/env bash
# Submit the benchmark to OAR, respecting the Grid'5000 usage policy.
#
#   RESULTS=/srv/storage/evergreen@storage1.sophia.grid5000.fr/$USER/geoparser-bench ./scripts/g5k/submit.sh
#   RESULTS=/srv/storage/evergreen@storage1.sophia.grid5000.fr/$USER/geoparser-bench WALLTIME=4:00 LIMIT=50 ./scripts/g5k/submit.sh
#   RESULTS=/srv/storage/evergreen@storage1.sophia.grid5000.fr/$USER/geoparser-bench CORPORA=all PIPELINES=prior,trim PHASES=resolution ./scripts/g5k/submit.sh
#   GEOPARSER_GAZETTEER_BUILD_MEMORY_LIMIT_MB=6144 RESULTS=/srv/storage/evergreen@storage1.sophia.grid5000.fr/$USER/geoparser-bench ./scripts/g5k/submit.sh
#
# PIPELINES accepts benchmark pipeline names, CORPORA accepts registered
# corpus names or "all", and PHASES accepts recognition and/or resolution.
# Each value is comma-separated and passed to the benchmark CLI on the node.
#
# Policy notes this script encodes:
#   * one GPU and four CPU cores on one host -- the workload is single-process
#     inference, so more would sit idle while still being unavailable to others
#   * an explicit, modest walltime rather than the maximum
#   * a refusal to submit when a job of this name is already queued or running,
#     because duplicate and speculative reservations are what the policy
#     specifically prohibits
#
# Check conformance before and after:  usagepolicycheck -t
# RESULTS is required so checkpoints and large artifacts stay on durable
# shared storage instead of consuming a frontend's home quota.

set -euo pipefail

JOB_NAME="${JOB_NAME:-geoparser-bench}"
WALLTIME="${WALLTIME:-2:00}"
GPUS="${GPUS:-1}"
CORES="${CORES:-4}"
REPO_ROOT="${REPO_ROOT:-$HOME/geoparser}"
VENV="${VENV:-$HOME/geoparser-venv}"
: "${RESULTS:?Set RESULTS to a durable shared filesystem path (for example Grid5000 Evergreen)}"
CLUSTER="${CLUSTER:-}"
if [[ -n "$CLUSTER" && ! "$CLUSTER" =~ ^[a-z0-9-]+$ ]]; then
    echo "CLUSTER must be a simple Grid5000 cluster name" >&2
    exit 2
fi
# besteffort takes GPUs that would otherwise sit idle, and gives them straight
# back when someone with a claim on them asks. The job is killed rather than
# queued behind, which is why it checkpoints per chunk; paired with OAR's
# idempotent type it is resubmitted automatically and resumes where it stopped.
# Some sites grant a user no other access to a GPU cluster, and OAR says so:
# "You can only access the required resources in besteffort."
QUEUE="${QUEUE:-default}"
PIPELINES="${PIPELINES:-}"
# Comma-separated corpus names from scripts/benchmark/corpora.py, or "all".
CORPORA="${CORPORA:-}"

if [[ -n "$CORPORA" && ! "$CORPORA" =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]]; then
    echo "CORPORA must be a comma-separated list of corpus names, or all" >&2
    exit 2
fi

# Names are checked by the benchmark CLI itself; this only keeps the value
# safe to splice into the job's command line.
if [[ -n "$PIPELINES" && ! "$PIPELINES" =~ ^[a-z0-9.-]+(,[a-z0-9.-]+)*$ ]]; then
    echo "PIPELINES must be a comma-separated list of pipeline names" >&2
    exit 2
fi

# recognition and/or resolution; empty runs both.
PHASES="${PHASES:-}"
if [[ -n "$PHASES" && ! "$PHASES" =~ ^(recognition|resolution)(,(recognition|resolution))?$ ]]; then
    echo "PHASES must be recognition, resolution, or both, comma-separated" >&2
    exit 2
fi

GAZETTEER_MEMORY_LIMIT_MB="${GEOPARSER_GAZETTEER_BUILD_MEMORY_LIMIT_MB:-}"
if [[ -n "$GAZETTEER_MEMORY_LIMIT_MB" && ! "$GAZETTEER_MEMORY_LIMIT_MB" =~ ^[1-9][0-9]*$ ]]; then
    echo "GEOPARSER_GAZETTEER_BUILD_MEMORY_LIMIT_MB must be a positive integer number of MiB" >&2
    exit 2
fi

mkdir -p "$RESULTS/logs"

if ! oarstat_json="$(oarstat -u --json)"; then
    echo "Could not inspect queued/running OAR jobs; refusing to submit." >&2
    exit 1
fi

if ! existing="$(BENCHMARK_JOB_NAME="$JOB_NAME" python3 -c '
import json, os, sys

try:
    jobs = json.load(sys.stdin)
except Exception as error:
    print(f"Could not parse oarstat JSON: {error}", file=sys.stderr)
    sys.exit(1)

if isinstance(jobs, dict):
    jobs = list(jobs.values())
elif not isinstance(jobs, list):
    print("oarstat JSON must be a list or object of jobs", file=sys.stderr)
    sys.exit(1)

for job in jobs:
    if not isinstance(job, dict):
        print("oarstat JSON contains a non-object job", file=sys.stderr)
        sys.exit(1)
    if not isinstance(job.get("name"), str) or not isinstance(job.get("state"), str):
        print("oarstat JSON contains a job without a name or state", file=sys.stderr)
        sys.exit(1)
    if (
        job["name"] == os.environ["BENCHMARK_JOB_NAME"]
        and job["state"] in {"Waiting", "Running", "Launching", "toLaunch"}
    ):
        job_id = job.get("Job_Id") or job.get("id")
        if job_id is None:
            print("Matching OAR job has no identifier", file=sys.stderr)
            sys.exit(1)
        print(job_id)
' <<< "$oarstat_json")"; then
    echo "Could not validate queued/running OAR jobs; refusing to submit." >&2
    exit 1
fi

if [[ -n "$existing" ]]; then
    echo "A '${JOB_NAME}' job is already queued or running: ${existing}" >&2
    echo "Submitting another would be a duplicate reservation. Use 'oarstat -u'," >&2
    echo "or 'oardel ${existing}' if it is stale." >&2
    exit 1
fi

RESOURCES="host=1/gpu=${GPUS}/core=${CORES},walltime=${WALLTIME}"
OPTIONS=()
if [[ -n "$CLUSTER" ]]; then
    OPTIONS+=(-p "cluster='${CLUSTER}'")
fi
case "$CLUSTER" in
    drac|ecotaxe|grouille|sirius|chicoree|neowise)
        # These accelerator clusters are marked exotic in Grid5000's hardware catalog.
        OPTIONS+=(-t exotic)
        ;;
esac
# How long before the walltime OAR sends the checkpoint signal. besteffort
# jobs are capped at a minute, which is enough: the run's own progress is
# already on disk after every chunk, so this signal is a courtesy rather than
# the thing that saves the work.
CHECKPOINT_SECONDS=600
if [[ "$QUEUE" == "besteffort" ]]; then
    # idempotent tells OAR to resubmit the job when it is preempted. That is
    # only safe because the work resumes from its checkpoint rather than
    # starting over, so a resubmission costs the current chunk and no more.
    OPTIONS+=(-q besteffort -t besteffort -t idempotent)
    CHECKPOINT_SECONDS=60
fi

echo "submitting: oarsub -l ${RESOURCES} ${OPTIONS[*]-}"
shell_quote() {
    local value=${1//\'/\'\\\'\'}
    printf "'%s'" "$value"
}

WORKER_COMMAND="REPO_ROOT=$(shell_quote "$REPO_ROOT")"
WORKER_COMMAND+=" VENV=$(shell_quote "$VENV")"
WORKER_COMMAND+=" RESULTS=$(shell_quote "$RESULTS")"
WORKER_COMMAND+=" LIMIT=$(shell_quote "${LIMIT:-}")"
WORKER_COMMAND+=" DEVICE=$(shell_quote "${DEVICE:-auto}")"
WORKER_COMMAND+=" CHUNK_SIZE=$(shell_quote "${CHUNK_SIZE:-5}")"
WORKER_COMMAND+=" MIN_SIMILARITY=$(shell_quote "${MIN_SIMILARITY:-0.0}")"
WORKER_COMMAND+=" PIPELINES=$(shell_quote "$PIPELINES")"
WORKER_COMMAND+=" CORPORA=$(shell_quote "$CORPORA")"
WORKER_COMMAND+=" PHASES=$(shell_quote "$PHASES")"
WORKER_COMMAND+=" GEOPARSER_GAZETTEER_BUILD_MEMORY_LIMIT_MB=$(shell_quote "$GAZETTEER_MEMORY_LIMIT_MB")"
WORKER_COMMAND+=" bash $(shell_quote "$REPO_ROOT/scripts/g5k/run_benchmark.sh")"

oarsub \
    -n "$JOB_NAME" \
    -l "$RESOURCES" \
    "${OPTIONS[@]}" \
    -O "$RESULTS/logs/%jobid%.out" \
    -E "$RESULTS/logs/%jobid%.err" \
    --checkpoint "$CHECKPOINT_SECONDS" \
    --signal 12 \
    -- "$WORKER_COMMAND"

echo
echo "Watch it with:   oarstat -u"
echo "Logs:            $RESULTS/logs/"
echo "Cancel with:     oardel <job id>"
