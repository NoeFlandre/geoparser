from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ROOT = Path("/workspace/geoparser")
RUN_ROOT = Path("/workspace/panx-run/full")
CACHE_DIR = Path("/workspace/panx-cache")
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
OUTPUT_DIR = RUN_ROOT / "report"
STATE_PATH = RUN_ROOT / "upload-state.json"
STOP_FILE = RUN_ROOT / "UPLOAD_DONE"
RESULTS_REPO = "NoeFlandre/geoparser-benchmark-results"
RUN_ID = "2026-10-02-cd0c750-transformers-5-18-cpu"
REMOTE_ROOT = f"runs/panx/{RUN_ID}/checkpoints"
POLL_SECONDS = 15


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_state(value: dict) -> None:
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(STATE_PATH)


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"verified": {}, "pending": None, "last_commit": None}
    value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    value.setdefault("verified", {})
    value.setdefault("pending", None)
    value.setdefault("last_commit", None)
    return value


def _response_facts(
    error: BaseException,
) -> tuple[int | None, int | None, str | None, str | None]:
    response = getattr(error, "response", None)
    request = getattr(response, "request", None) if response is not None else None
    status = getattr(response, "status_code", None) if response is not None else None
    headers = getattr(response, "headers", {}) if response is not None else {}
    retry_after = headers.get("Retry-After") if headers else None
    try:
        retry_seconds = max(1, int(retry_after)) if retry_after else None
    except (TypeError, ValueError):
        retry_seconds = None
    method = getattr(request, "method", None) if request is not None else None
    url = getattr(request, "url", None) if request is not None else None
    host = urlsplit(str(url)).hostname if url else None
    return status, retry_seconds, host, method


def _retry_delay(error: BaseException) -> int | None:
    status, retry_after, host, method = _response_facts(error)
    print(
        f"remote_request_failed status={status} host={host or 'unknown'} "
        f"method={method or 'unknown'}",
        flush=True,
    )
    if status == 429:
        return retry_after or 60
    if status is None or status >= 500:
        return 60
    return None


def _verify_pending(api: HfApi, state: dict) -> None:
    pending = state.get("pending")
    if not pending:
        return
    revision = pending["revision"]
    for item in pending["items"]:
        downloaded = api.hf_hub_download(
            repo_id=RESULTS_REPO,
            filename=item["remote_path"],
            repo_type="dataset",
            revision=revision,
            cache_dir="/workspace/hf-cache/hub",
        )
        actual = file_sha256(Path(downloaded))
        if actual != item["sha256"]:
            raise RuntimeError(f"Remote SHA-256 mismatch for {item['remote_path']}")
        state["verified"][item["local_path"]] = item["sha256"]
        print(f"verified {item['remote_path']} sha256={actual}", flush=True)
    state["last_commit"] = revision
    state["pending"] = None
    write_state(state)


def _changed_files(state: dict) -> list[tuple[Path, str, str]]:
    changed = []
    if not CHECKPOINT_DIR.exists():
        return changed
    pending_paths = {
        item["local_path"] for item in (state.get("pending") or {}).get("items", [])
    }
    for path in sorted(CHECKPOINT_DIR.rglob("*.json")):
        if path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(CHECKPOINT_DIR).as_posix()
        if relative in pending_paths:
            continue
        digest = file_sha256(path)
        if state["verified"].get(relative) != digest:
            changed.append((path, relative, digest))
    return changed


def _publish_changed(
    api: HfApi, state: dict, changed: list[tuple[Path, str, str]]
) -> None:
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{REMOTE_ROOT}/{relative}",
            path_or_fileobj=path,
        )
        for path, relative, _ in changed
    ]
    commit = api.create_commit(
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        revision="main",
        operations=operations,
        commit_message=f"PAN-X checkpoint sync: {len(changed)} completed files",
        num_threads=2,
    )
    state["pending"] = {
        "revision": commit.oid,
        "items": [
            {
                "local_path": relative,
                "remote_path": f"{REMOTE_ROOT}/{relative}",
                "sha256": digest,
            }
            for _, relative, digest in changed
        ],
    }
    write_state(state)
    print(f"uploaded commit={commit.oid} files={len(changed)}", flush=True)


def publisher(stop_event: threading.Event, error_box: list[BaseException]) -> None:
    api = HfApi()
    state = load_state()
    while True:
        try:
            _verify_pending(api, state)
            changed = _changed_files(state)
            if changed:
                _publish_changed(api, state, changed)
                _verify_pending(api, state)
        except Exception as error:  # noqa: BLE001 - report every uploader failure and stop evaluation
            delay = _retry_delay(error)
            if delay is None:
                error_box.append(error)
                print(
                    f"checkpoint_publisher_stopped error={type(error).__name__}",
                    flush=True,
                )
                return
            print(f"checkpoint_sync_retry_in={delay}s", flush=True)
            stop_event.wait(delay)
            continue
        if (
            stop_event.is_set()
            and not state.get("pending")
            and not _changed_files(state)
        ):
            print(
                f"checkpoint_sync_complete commit={state.get('last_commit')}",
                flush=True,
            )
            return
        if stop_event.wait(POLL_SECONDS):
            continue


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)

    uploader_stop = threading.Event()
    uploader_errors: list[BaseException] = []
    upload_thread = threading.Thread(
        target=publisher,
        args=(uploader_stop, uploader_errors),
        name="verified-hub-checkpoint-publisher",
        daemon=True,
    )
    upload_thread.start()

    command = [
        sys.executable,
        "-m",
        "scripts.panx_benchmark",
        "--cache-dir",
        str(CACHE_DIR),
        "--checkpoint-dir",
        str(CHECKPOINT_DIR),
        "--output-dir",
        str(OUTPUT_DIR),
    ]
    eval_env = os.environ.copy()
    eval_env["HF_HUB_OFFLINE"] = "1"
    eval_env["HF_DATASETS_OFFLINE"] = "1"
    eval_env["HF_XET_CACHE"] = "/workspace/hf-cache/xet"
    eval_env["HF_HUB_CACHE"] = "/workspace/hf-cache/hub"
    print(f"evaluation_started run_id={RUN_ID}", flush=True)
    print("remote_sync=per-checkpoint-with-exact-revision-readback", flush=True)

    process = subprocess.Popen(command, cwd=REPO_ROOT, env=eval_env)
    try:
        while process.poll() is None:
            if uploader_errors:
                print(
                    "stopping evaluation because remote checkpoint verification failed",
                    flush=True,
                )
                process.send_signal(signal.SIGINT)
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("interrupt_received; preserving completed local checkpoints", flush=True)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    return_code = process.wait()
    uploader_stop.set()
    upload_thread.join()
    if uploader_errors:
        return 2
    if return_code != 0:
        print(f"evaluation_exit_code={return_code}", flush=True)
        return return_code
    print("evaluation_complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
