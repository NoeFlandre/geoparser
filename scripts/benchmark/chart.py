"""
Draw the dataset card's results chart as a self-contained SVG.

A grouped horizontal bar chart: one group per corpus, one bar per pipeline,
every bar labelled with its value and named in a ``<title>`` so hovering it
shows what it is. The SVG carries its own light and dark colours, so it reads
on either Hugging Face theme without a plotting dependency.

Colours are the first categorical slots of the dataviz reference palette,
validated for colour-vision deficiency on both surfaces. A pipeline keeps its
slot whichever pipelines ran on a corpus: colour follows the entity.
"""

from __future__ import annotations

import math
import typing as t
from xml.sax.saxutils import escape

# Pipelines in the order they take colour slots. A pipeline outside this list
# is appended after them, so a new one never repaints the existing ones.
PIPELINE_ORDER = (
    "upstream",
    "swapped",
    "hybrid",
    "prior",
    "trim",
    "population",
    "population-0.05",
    "population-0.2",
)

SERIES_COLORS = {
    "light": (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#4a3aa7",
        "#e34948",
    ),
    "dark": (
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
}
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
INK = {"light": ("#0b0b0b", "#52514e"), "dark": ("#ffffff", "#c3c2b7")}
GRID = {"light": "#e4e3df", "dark": "#3a3a37"}

WIDTH = 760
LABEL_WIDTH = 150
VALUE_WIDTH = 48
BAR_HEIGHT = 10
BAR_GAP = 2
GROUP_GAP = 16
TOP = 64
BOTTOM = 28
# A log axis spans 1 km to the longest possible error, half the circumference.
LOG_MAX_KM = 20_040.0
LOG_TICKS_KM = (1, 10, 100, 1_000, 10_000)
UNIT_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _position(value: float, scale: str) -> float:
    """Where a value sits along the axis, as a fraction of its length."""
    if scale == "log":
        return math.log10(max(value, 1.0)) / math.log10(LOG_MAX_KM)
    return value


def _format(value: float, scale: str) -> str:
    """A value as printed beside its bar and in its tooltip."""
    return f"{value:.0f} km" if scale == "log" else f"{value:.3f}"


def _ticks(scale: str) -> list[tuple[float, str]]:
    """Axis ticks as (value, label)."""
    if scale == "log":
        return [(tick, f"{tick} km") for tick in LOG_TICKS_KM]
    return [(tick, f"{tick:g}") for tick in UNIT_TICKS]


def _style() -> str:
    """CSS for both themes; dark mode has its own steps, not an inversion."""

    def rules(mode: str) -> str:
        primary, secondary = INK[mode]
        series = " ".join(
            f".s{slot}{{fill:{color}}}"
            for slot, color in enumerate(SERIES_COLORS[mode])
        )
        return (
            f".bg{{fill:{SURFACE[mode]}}} .t1{{fill:{primary}}} "
            f".t2{{fill:{secondary}}} .grid{{stroke:{GRID[mode]}}} {series}"
        )

    return (
        "text{font:12px system-ui,-apple-system,Segoe UI,sans-serif}"
        ".title{font-size:14px;font-weight:600} .v{font-size:10px} .bar:hover{opacity:.8} "
        + rules("light")
        + " @media (prefers-color-scheme: dark){"
        + rules("dark")
        + "}"
    )


def _pipelines(latest: t.Mapping[tuple[str, str], t.Any]) -> list[str]:
    """Pipelines present, in their fixed colour order."""
    present = {pipeline for _, pipeline in latest}
    known = [name for name in PIPELINE_ORDER if name in present]
    return known + sorted(present - set(known))


def render_bar_chart(
    latest: t.Mapping[tuple[str, str], t.Mapping[str, t.Any]],
    metric: str,
    title: str,
    *,
    corpora: t.Sequence[str] | None = None,
    scale: str = "unit",
) -> str:
    """
    Return a grouped horizontal bar chart of one metric, as SVG.

    Args:
        latest: (corpus, pipeline) to its result row
        metric: The row key to plot
        title: The chart's heading
        corpora: Group order; defaults to first appearance in ``latest``
        scale: ``unit`` for a score in [0, 1], ``log`` for a distance in km

    Returns:
        A standalone SVG document
    """
    pipelines = _pipelines(latest)
    slots = {name: index for index, name in enumerate(PIPELINE_ORDER)}
    for name in pipelines:
        slots.setdefault(name, len(slots))
    groups = list(corpora or dict.fromkeys(corpus for corpus, _ in latest))

    group_height = len(pipelines) * (BAR_HEIGHT + BAR_GAP) - BAR_GAP
    height = TOP + len(groups) * (group_height + GROUP_GAP) + BOTTOM
    plot = WIDTH - LABEL_WIDTH - VALUE_WIDTH

    def x(value: float) -> float:
        return LABEL_WIDTH + plot * _position(value, scale)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" '
        f'aria-label="{escape(title)}">',
        f"<style>{_style()}</style>",
        f'<rect class="bg" width="{WIDTH}" height="{height}" rx="8"/>',
        f'<text class="title t1" x="16" y="24">{escape(title)}</text>',
    ]
    legend_x = 16.0
    for name in pipelines:
        parts.append(
            f'<rect class="s{slots[name]}" x="{legend_x}" y="36" width="10" '
            f'height="10" rx="2"/><text class="t2" x="{legend_x + 14}" '
            f'y="45">{escape(name)}</text>'
        )
        legend_x += 24 + 7.5 * len(name)

    plot_bottom = height - BOTTOM
    for tick, label in _ticks(scale):
        parts.append(
            f'<line class="grid" x1="{x(tick)}" y1="{TOP - 6}" x2="{x(tick)}" '
            f'y2="{plot_bottom}" stroke-width="1"/><text class="t2" '
            f'x="{x(tick)}" y="{plot_bottom + 16}" '
            f'text-anchor="middle">{label}</text>'
        )

    y = float(TOP)
    for corpus in groups:
        rows = [latest.get((corpus, name)) for name in pipelines]
        language = next((row["language"] for row in rows if row), "")
        parts.append(
            f'<text class="t1" x="{LABEL_WIDTH - 10}" '
            f'y="{y + group_height / 2 + 4}" text-anchor="end">{escape(corpus)}'
            f'<tspan class="t2"> {escape(language)}</tspan></text>'
        )
        for name, row in zip(pipelines, rows, strict=True):
            if row is not None and row.get(metric) is not None:
                value = float(row[metric])
                shown = _format(value, scale)
                width = max(x(value) - LABEL_WIDTH, 1)
                parts.append(
                    f'<rect class="bar s{slots[name]}" x="{LABEL_WIDTH}" '
                    f'y="{y}" width="{width:.1f}" '
                    f'height="{BAR_HEIGHT}" rx="2"><title>{escape(corpus)} · '
                    f"{escape(name)}: {shown}</title></rect>"
                    f'<text class="t2 v" x="{LABEL_WIDTH + width + 4:.1f}" '
                    f'y="{y + 9}">{shown}</text>'
                )
            y += BAR_HEIGHT + BAR_GAP
        y += GROUP_GAP - BAR_GAP
    parts.append("</svg>")
    return "".join(parts)
