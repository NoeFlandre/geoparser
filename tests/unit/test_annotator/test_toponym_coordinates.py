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
def test_coordinate_conversion_returns_none_for_missing_geometry():
    """Features without geometry omit coordinates."""
    feature = SimpleNamespace(geometry=None, crs="EPSG:4326")

    assert ToponymRepository._get_wgs84_coordinates(cast(Any, feature)) == (None, None)


@pytest.mark.unit
def test_coordinate_conversion_preserves_wgs84_centroid_order():
    """WGS84 centroids are returned in latitude-longitude order."""
    feature = SimpleNamespace(
        geometry=SimpleNamespace(centroid=SimpleNamespace(x=2.0, y=48.0)),
        crs="EPSG:4326",
    )

    assert ToponymRepository._get_wgs84_coordinates(cast(Any, feature)) == (48.0, 2.0)


@pytest.mark.unit
def test_coordinate_conversion_transforms_projected_centroid(monkeypatch):
    """Projected centroids are transformed to WGS84 and reordered."""
    from geoparser.annotator.db.crud import toponym

    transformer = Mock()
    transformer.transform.return_value = (2.0, 48.0)
    monkeypatch.setattr(toponym.Transformer, "from_crs", Mock(return_value=transformer))
    feature = SimpleNamespace(
        geometry=SimpleNamespace(centroid=SimpleNamespace(x=100.0, y=200.0)),
        crs="EPSG:2056",
    )

    assert ToponymRepository._get_wgs84_coordinates(cast(Any, feature)) == (48.0, 2.0)
    transformer.transform.assert_called_once_with(100.0, 200.0)


@pytest.mark.unit
def test_coordinate_conversion_propagates_unexpected_errors():
    """Programming errors do not get silently converted into missing points."""
    feature = SimpleNamespace(
        geometry=_BrokenGeometry(RuntimeError("unexpected failure")),
        crs="EPSG:4326",
    )

    with pytest.raises(RuntimeError, match="unexpected failure"):
        ToponymRepository._get_wgs84_coordinates(cast(Any, feature))
