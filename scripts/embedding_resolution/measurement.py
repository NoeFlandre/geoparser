"""Latency and memory measurements, kept apart from the scores they describe."""

from __future__ import annotations

import sys
import time
import typing as t


def timed(function: t.Callable[[], t.Any]) -> tuple[t.Any, float]:
    """
    Run one call and return its result with the wall-clock seconds it took.

    Args:
        function: A zero-argument callable; bind arguments with a lambda or partial

    Returns:
        The call's result and its duration in seconds, never negative
    """
    started = time.perf_counter()
    result = function()
    return result, max(0.0, time.perf_counter() - started)


def peak_rss_bytes() -> int:
    """
    Return this process's peak resident set size in bytes.

    ``getrusage`` reports bytes on macOS and kibibytes elsewhere, so the unit is
    normalized here. The benchmark protocol rejects a zero value because a zero
    would read as a measurement rather than as an unavailable one.

    Returns:
        The peak resident set size in bytes
    """
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak) * 1024
