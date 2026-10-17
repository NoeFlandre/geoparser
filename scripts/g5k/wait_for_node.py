#!/usr/bin/env python3
"""Wait for a deployed Grid'5000 node to accept an OAR SSH command."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Callable

ProbeResult = tuple[bool, str]
ReadinessProbe = Callable[[float], ProbeResult]

DEFAULT_TIMEOUT_SECONDS = 15 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 10
DEFAULT_PROBE_TIMEOUT_SECONDS = 10


def wait_until_ready(
    probe: ReadinessProbe,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, int, str | None]:
    """Retry a readiness probe until it succeeds or the monotonic deadline passes."""
    _validate_wait_options(timeout_seconds, poll_interval_seconds)

    deadline = monotonic() + timeout_seconds
    attempts = 0
    last_error: str | None = None
    while (remaining := deadline - monotonic()) > 0:
        attempts += 1
        ready, error = probe(remaining)
        if ready:
            if deadline - monotonic() > 0:
                return True, attempts, None
            return False, attempts, last_error
        if error:
            last_error = error
        _sleep_before_next_probe(
            deadline=deadline,
            poll_interval_seconds=poll_interval_seconds,
            monotonic=monotonic,
            sleep=sleep,
        )

    return False, attempts, last_error


def _sleep_before_next_probe(
    *,
    deadline: float,
    poll_interval_seconds: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    """Sleep for at most the polling interval and the remaining deadline."""
    remaining = deadline - monotonic()
    if remaining > 0:
        sleep(min(poll_interval_seconds, remaining))


def _validate_wait_options(
    timeout_seconds: float, poll_interval_seconds: float
) -> None:
    """Require positive timeout and polling durations."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        message = "timeout_seconds must be positive"
        raise ValueError(message)
    if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
        message = "poll_interval_seconds must be positive"
        raise ValueError(message)


def _positive_seconds(value: str) -> float:
    """Parse a positive duration for the command-line interface."""
    try:
        seconds = float(value)
    except ValueError as error:
        message = "must be a number of seconds"
        raise argparse.ArgumentTypeError(message) from error
    if not math.isfinite(seconds) or seconds <= 0:
        message = "must be positive"
        raise argparse.ArgumentTypeError(message)
    return seconds


def _timeout_detail(host: str, attempts: int, last_error: str | None) -> str:
    """Format the final timeout message for a failed readiness wait."""
    detail = f" Last error: {last_error}" if last_error else ""
    return f"Timed out waiting for {host} after {attempts} probe(s).{detail}"


def _oarsh_probe(host: str, timeout_seconds: float) -> ProbeResult:
    """Return whether a harmless `oarsh host true` probe succeeded."""
    probe_timeout = min(timeout_seconds, DEFAULT_PROBE_TIMEOUT_SECONDS)
    try:
        result = subprocess.run(
            ["oarsh", host, "true"],
            capture_output=True,
            check=False,
            text=True,
            timeout=probe_timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"oarsh probe timed out after {probe_timeout:g}s"
    except OSError as error:
        return False, str(error)

    if result.returncode == 0:
        return True, ""
    return (
        False,
        result.stderr.strip() or f"oarsh exited with status {result.returncode}",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the readiness wait."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", help="Grid'5000 node name, for example hydra-1")
    parser.add_argument(
        "--timeout",
        type=_positive_seconds,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="maximum wait in seconds (default: 900)",
    )
    parser.add_argument(
        "--interval",
        type=_positive_seconds,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="seconds between probes (default: 10)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if shutil.which("oarsh") is None:
        print("oarsh is not available on this frontend", file=sys.stderr)
        return 2

    print(
        f"Waiting up to {args.timeout:g}s for {args.host} to accept oarsh...",
        flush=True,
    )
    ready, attempts, last_error = wait_until_ready(
        lambda remaining: _oarsh_probe(args.host, remaining),
        timeout_seconds=args.timeout,
        poll_interval_seconds=args.interval,
    )
    if ready:
        print(f"{args.host} is ready after {attempts} probe(s).")
        return 0

    print(
        _timeout_detail(args.host, attempts, last_error),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
