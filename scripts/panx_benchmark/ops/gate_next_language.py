from __future__ import annotations

import ctypes
import hashlib
import json
import os
import signal
import struct
import sys
import time
from pathlib import Path

PID = int(sys.argv[1])
CHECKPOINT_ROOT = Path("/workspace/panx-run/full/checkpoints")
SNAPSHOT_ID = "62a90f96e0db98c1d881feb316bd8a2205c8a1665a152941b8b319abd4079653"
WATCH_DIR = CHECKPOINT_ROOT / SNAPSHOT_ID / "languages" / "gliner2_multi"
STATE_PATH = Path("/workspace/panx-run/full/upload-state.json")
IN_MOVED_TO = 0x00000080

libc = ctypes.CDLL(None, use_errno=True)
fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
if fd < 0:
    raise OSError(ctypes.get_errno(), "inotify_init1 failed")
watch = libc.inotify_add_watch(fd, os.fsencode(WATCH_DIR), IN_MOVED_TO)
if watch < 0:
    raise OSError(ctypes.get_errno(), f"inotify_add_watch failed for {WATCH_DIR}")

print(f"checkpoint_gate_ready pid={PID} watch={WATCH_DIR}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


while is_alive(PID):
    try:
        data = os.read(fd, 4096)
    except BlockingIOError:
        time.sleep(0.1)
        continue
    offset = 0
    while offset < len(data):
        _, mask, _, name_length = struct.unpack_from("iIII", data, offset)
        offset += struct.calcsize("iIII")
        name = data[offset : offset + name_length].split(b"\0", 1)[0].decode()
        offset += name_length
        if not (mask & IN_MOVED_TO) or not name.endswith(".json"):
            continue
        path = WATCH_DIR / name
        if not path.is_file():
            continue
        relative = path.relative_to(CHECKPOINT_ROOT).as_posix()
        expected = sha256(path)
        try:
            os.kill(PID, signal.SIGSTOP)
        except ProcessLookupError:
            break
        print(f"evaluator_paused path={relative} sha256={expected}", flush=True)
        while is_alive(PID):
            try:
                state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(1)
                continue
            if state.get("verified", {}).get(relative) == expected:
                try:
                    os.kill(PID, signal.SIGCONT)
                except ProcessLookupError:
                    break
                print(
                    f"evaluator_resumed path={relative} sha256={expected} "
                    f"hf_commit={state.get('last_commit')}",
                    flush=True,
                )
                break
            time.sleep(1)
        if not is_alive(PID):
            break

os.close(fd)
print("checkpoint_gate_exit", flush=True)
