"""
Tests for the results chart on the dataset card.

The chart is a static SVG, so what matters is that every value is drawn,
labelled and named for a hover tooltip, and that colour follows the pipeline.
"""

import xml.etree.ElementTree as ET

from scripts.benchmark.chart import SERIES_COLORS, render_bar_chart

SVG = "{http://www.w3.org/2000/svg}"

LATEST = {
    ("geovirus", "upstream"): {"language": "en", "accuracy_at_161km": 0.834},
    ("geovirus", "hybrid"): {"language": "en", "accuracy_at_161km": 0.834},
    ("newseye-fi", "hybrid"): {"language": "fi", "accuracy_at_161km": 0.313},
}


def parse(svg):
    """Parse the SVG, failing the test if it is not well-formed."""
    return ET.fromstring(svg)


class TestRenderBarChart:
    """A grouped horizontal bar chart: a group per corpus, a bar per pipeline."""

    def test_is_well_formed_svg(self):
        """The card embeds it as an image, so it must parse."""
        root = parse(render_bar_chart(LATEST, "accuracy_at_161km", "Acc"))

        assert root.tag == f"{SVG}svg"

    def test_draws_one_bar_per_result_with_a_tooltip(self):
        """Every score is a bar that names itself on hover."""
        root = parse(render_bar_chart(LATEST, "accuracy_at_161km", "Acc"))

        titles = [t.text for t in root.iter(f"{SVG}title")]
        assert "geovirus · upstream: 0.834" in titles
        assert "newseye-fi · hybrid: 0.313" in titles
        assert len(root.findall(f".//{SVG}rect[@class='bar s2']")) == 2

    def test_labels_every_bar_with_its_value(self):
        """Values are printed, so identity and value never rest on colour."""
        svg = render_bar_chart(LATEST, "accuracy_at_161km", "Acc")

        assert svg.count(">0.834<") == 2
        assert ">0.313<" in svg

    def test_colour_follows_the_pipeline(self):
        """A pipeline keeps its slot whichever pipelines ran on a corpus."""
        svg = render_bar_chart(LATEST, "accuracy_at_161km", "Acc")

        assert "bar s0" in svg and "bar s2" in svg
        assert len(SERIES_COLORS["light"]) >= 4

    def test_has_a_legend_naming_each_pipeline(self):
        """With several series a legend is always present."""
        svg = render_bar_chart(LATEST, "accuracy_at_161km", "Acc")

        assert ">upstream<" in svg and ">hybrid<" in svg

    def test_supports_dark_mode(self):
        """Dark mode has its own validated steps, not an inversion."""
        svg = render_bar_chart(LATEST, "accuracy_at_161km", "Acc")

        assert "prefers-color-scheme: dark" in svg


ERRORS = {
    ("geovirus", "hybrid"): {"language": "en", "median_error_km": 30.5},
    ("newseye-fi", "hybrid"): {"language": "fi", "median_error_km": 1200.0},
}


class TestLogScale:
    """Distance errors span four orders of magnitude."""

    def test_ticks_are_powers_of_ten_in_km(self):
        """A log axis labels decades, in the unit of the metric."""
        svg = render_bar_chart(ERRORS, "median_error_km", "Err", scale="log")

        for tick in (">1 km<", ">10 km<", ">100 km<", ">1000 km<"):
            assert tick in svg

    def test_bar_length_is_logarithmic(self):
        """A 40x larger error is a bit over one decade longer, not 40x."""
        root = parse(render_bar_chart(ERRORS, "median_error_km", "Err", scale="log"))
        widths = [
            float(r.get("width"))
            for r in root.iter(f"{SVG}rect")
            if "bar" in (r.get("class") or "")
        ]

        assert 1.5 < widths[1] / widths[0] < 2.5

    def test_values_keep_their_unit(self):
        """A km value is printed as km, rounded to whole kilometres."""
        svg = render_bar_chart(ERRORS, "median_error_km", "Err", scale="log")

        assert ">31 km<" in svg or ">30 km<" in svg
        assert ">1200 km<" in svg
