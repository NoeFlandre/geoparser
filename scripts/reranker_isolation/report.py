"""Compare each reranker with the no-reranker baseline on the same examples.

Every system is paired with the baseline by example ID. Gain is the accuracy
difference on those paired examples. Its interval comes from a paired document
bootstrap, which takes the difference on each shared draw before it takes
percentiles. Loading time and steady inference time are kept apart, and an
unavailable memory reading stays unavailable rather than becoming zero.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from scripts.reranker_isolation.ranking import Decision


def _non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        msg = f"{name} must be finite and non-negative, got {value}"
        raise ValueError(msg)


@dataclass(frozen=True)
class Outcome:
    """One system's decision for one eligible example, and its steady time."""

    example_id: str
    system: str
    gold_id: str
    decision: Decision
    seconds: float

    def __post_init__(self) -> None:
        _non_negative(self.seconds, "seconds")

    @property
    def correct(self) -> bool:
        """Resolved, and resolved to the gold canonical ID."""
        return self.decision.status == "resolved" and (
            self.decision.identifier == self.gold_id
        )


@dataclass(frozen=True)
class Timing:
    """Separate measurements for one system, in seconds and bytes."""

    fetch_seconds: float
    load_seconds: float
    warmup_seconds: float
    steady_seconds: float
    peak_rss_bytes: int | None

    def __post_init__(self) -> None:
        _non_negative(self.fetch_seconds, "fetch_seconds")
        _non_negative(self.load_seconds, "load_seconds")
        _non_negative(self.warmup_seconds, "warmup_seconds")
        _non_negative(self.steady_seconds, "steady_seconds")
        if self.peak_rss_bytes is not None and self.peak_rss_bytes <= 0:
            msg = f"peak RSS must be positive or unavailable, got {self.peak_rss_bytes}"
            raise ValueError(msg)


@dataclass(frozen=True)
class SystemSummary:
    """Counts, gain against the baseline and timing for one system.

    The gain fields stay empty for the baseline itself.
    """

    system: str
    eligible: int
    correct: int
    resolved: int
    abstained: int
    invalid: int
    accuracy: float
    mean_steady_seconds: float
    timing: Timing
    extra_peak_rss_bytes: int | None
    gain: float | None = None
    gain_interval: tuple[float, float] | None = None
    fixes: int = 0
    regressions: int = 0


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    """Return the empirical inverse-CDF 2.5th and 97.5th percentiles.

    This is the same rule the public benchmark protocol uses.
    """
    if not values:
        msg = "cannot take percentiles of no values"
        raise ValueError(msg)
    ordered = sorted(values)
    size = len(ordered)
    return (
        ordered[max(0, (25 * size + 999) // 1000 - 1)],
        ordered[(975 * size + 999) // 1000 - 1],
    )


def paired_gain_interval(
    system: Sequence[bool], baseline: Sequence[bool], *, seed: int, resamples: int
) -> tuple[float, float]:
    """Bootstrap the accuracy difference, resampling whole examples.

    Each draw uses the same example indices for both systems. Its gain is the
    mean of the paired differences on those indices.

    Raises:
        ValueError: If the lists differ in length or ``resamples`` is below one.
    """
    _check_bootstrap_inputs(system, baseline, resamples)
    size = len(system)
    if size == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)  # noqa: S311 - seeded statistical resampling
    indices = range(size)
    gains = []
    for _ in range(resamples):
        draw = rng.choices(indices, k=size)
        gains.append(sum(int(system[i]) - int(baseline[i]) for i in draw) / size)
    return percentile_interval(gains)


def _check_bootstrap_inputs(
    system: Sequence[bool], baseline: Sequence[bool], resamples: int
) -> None:
    if len(system) != len(baseline):
        msg = "system and baseline must have the same length"
        raise ValueError(msg)
    if resamples < 1:
        msg = f"resamples must be at least 1, got {resamples}"
        raise ValueError(msg)


def _group_by_system(outcomes: Sequence[Outcome]) -> dict[str, dict[str, Outcome]]:
    grouped: dict[str, dict[str, Outcome]] = {}
    for row in outcomes:
        rows = grouped.setdefault(row.system, {})
        if row.example_id in rows:
            msg = f"duplicate outcome for {row.example_id!r} in {row.system!r}"
            raise ValueError(msg)
        rows[row.example_id] = row
    return grouped


def _check_pairing(
    grouped: Mapping[str, Mapping[str, Outcome]], baseline: str
) -> list[str]:
    if baseline not in grouped:
        msg = f"the baseline system {baseline!r} has no outcomes"
        raise ValueError(msg)
    reference = grouped[baseline]
    for system, rows in grouped.items():
        _check_system_rows(system, rows, reference)
    return sorted(reference)


def _check_system_rows(
    system: str, rows: Mapping[str, Outcome], reference: Mapping[str, Outcome]
) -> None:
    if set(rows) != set(reference):
        msg = f"paired examples differ for {system!r} and the baseline"
        raise ValueError(msg)
    for example_id, row in rows.items():
        if row.gold_id != reference[example_id].gold_id:
            msg = f"gold IDs differ for {example_id!r} in {system!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class _Pairing:
    gain: float
    interval: tuple[float, float]
    fixes: int
    regressions: int


def _pairing(
    flags: Sequence[bool], baseline_flags: Sequence[bool], *, seed: int, resamples: int
) -> _Pairing:
    fixes = sum(
        flag and not base for flag, base in zip(flags, baseline_flags, strict=True)
    )
    regressions = sum(
        base and not flag for flag, base in zip(flags, baseline_flags, strict=True)
    )
    gain = (sum(flags) - sum(baseline_flags)) / len(flags)
    interval = paired_gain_interval(
        flags, baseline_flags, seed=seed, resamples=resamples
    )
    return _Pairing(gain=gain, interval=interval, fixes=fixes, regressions=regressions)


def _extra_peak(timing: Timing, baseline_timing: Timing) -> int | None:
    if timing.peak_rss_bytes is None or baseline_timing.peak_rss_bytes is None:
        return None
    return timing.peak_rss_bytes - baseline_timing.peak_rss_bytes


@dataclass(frozen=True)
class _Plan:
    """What every system is compared on: shared IDs, baseline and bootstrap."""

    ids: list[str]
    baseline: str
    baseline_flags: list[bool]
    timings: Mapping[str, Timing]
    seed: int
    resamples: int

    @property
    def baseline_timing(self) -> Timing:
        return self.timings[self.baseline]


def summarize(
    outcomes: Sequence[Outcome],
    timings: Mapping[str, Timing],
    *,
    baseline: str,
    seed: int,
    resamples: int,
) -> dict[str, SystemSummary]:
    """Summarise every system against the baseline on the paired examples.

    Raises:
        ValueError: If there are no outcomes, the baseline is missing, systems do
            not cover the same examples, IDs repeat, gold IDs disagree, a timing
            is missing, or ``resamples`` is below one.
    """
    _check_summary_inputs(outcomes, resamples)
    grouped = _group_by_system(outcomes)
    ids = _check_pairing(grouped, baseline)
    _check_timings(grouped, timings)
    reference = grouped[baseline]
    plan = _Plan(
        ids=ids,
        baseline=baseline,
        baseline_flags=[reference[example_id].correct for example_id in ids],
        timings=timings,
        seed=seed,
        resamples=resamples,
    )
    return {
        system: _summarise_system(system, rows, plan)
        for system, rows in grouped.items()
    }


def _check_summary_inputs(outcomes: Sequence[Outcome], resamples: int) -> None:
    if not outcomes:
        msg = "summaries need at least one outcome"
        raise ValueError(msg)
    if resamples < 1:
        msg = f"resamples must be at least 1, got {resamples}"
        raise ValueError(msg)


def _check_timings(
    grouped: Mapping[str, Mapping[str, Outcome]], timings: Mapping[str, Timing]
) -> None:
    missing = sorted(set(grouped) - set(timings))
    if missing:
        msg = f"missing timing for {missing}"
        raise ValueError(msg)


def _summarise_system(
    system: str, rows: Mapping[str, Outcome], plan: _Plan
) -> SystemSummary:
    ordered = [rows[example_id] for example_id in plan.ids]
    flags = [row.correct for row in ordered]
    summary = _tally(system, ordered, flags, plan)
    if system == plan.baseline:
        return summary
    pairing = _pairing(
        flags, plan.baseline_flags, seed=plan.seed, resamples=plan.resamples
    )
    return replace(
        summary,
        gain=pairing.gain,
        gain_interval=pairing.interval,
        fixes=pairing.fixes,
        regressions=pairing.regressions,
    )


def _tally(
    system: str, ordered: Sequence[Outcome], flags: Sequence[bool], plan: _Plan
) -> SystemSummary:
    return SystemSummary(
        system=system,
        eligible=len(ordered),
        correct=sum(flags),
        resolved=_count_status(ordered, "resolved"),
        abstained=_count_status(ordered, "abstained"),
        invalid=_count_status(ordered, "invalid"),
        accuracy=sum(flags) / len(ordered),
        mean_steady_seconds=sum(row.seconds for row in ordered) / len(ordered),
        timing=plan.timings[system],
        extra_peak_rss_bytes=_extra_peak(plan.timings[system], plan.baseline_timing),
    )


def _count_status(ordered: Sequence[Outcome], status: str) -> int:
    return sum(row.decision.status == status for row in ordered)
