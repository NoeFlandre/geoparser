"""Tests for annotator gazetteer coordinate conversion."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from pyproj.exceptions import ProjError
from shapely.errors import GEOSException

from geoparser.annotator.db.crud.toponym import ToponymRepository


class _BrokenGeometry:
    """Geometry stand-in whose centroid access raises a chosen error."""

    def __init__(self, error: Exception):
        self.error = error

    @property
    def centroid(self):
        raise self.error


@pytest.mark.unit
def test_coordinate_conversion_returns_none_for_shapely_failure():
    """A geometry engine failure leaves a candidate without coordinates."""
    feature = SimpleNamespace(
        geometry=_BrokenGeometry(GEOSException("invalid geometry")),
        crs="EPSG:4326",
    )

    assert ToponymRepository._get_wgs84_coordinates(cast(Any, feature)) == (None, None)


@pytest.mark.unit
def test_coordinate_conversion_returns_none_for_projection_failure(monkeypatch):
    """A projection engine failure leaves a candidate without coordinates."""
    from geoparser.annotator.db.crud import toponym

    monkeypatch.setattr(
        toponym.Transformer,
        "from_crs",
        Mock(side_effect=ProjError("bad projection")),
    )
    feature = SimpleNamespace(
        geometry=SimpleNamespace(centroid=SimpleNamespace(x=1, y=2)),
        crs="invalid-crs",
    )

    assert ToponymRepository._get_wgs84_coordinates(cast(Any, feature)) == (None, None)


@pytest.mark.unit
def test_coordinate_conversion_propagates_unexpected_errors():
    """Programming errors do not get silently converted into missing points."""
    feature = SimpleNamespace(
        geometry=_BrokenGeometry(RuntimeError("unexpected failure")),
        crs="EPSG:4326",
    )

    with pytest.raises(RuntimeError, match="unexpected failure"):
        ToponymRepository._get_wgs84_coordinates(cast(Any, feature))
