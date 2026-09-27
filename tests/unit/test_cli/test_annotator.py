"""
Unit tests for geoparser/cli/annotator.py

Tests the annotator CLI wrapper.
"""

from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestAnnotatorCli:
    """Test annotator_cli() function."""

    @patch("geoparser.annotator.app.run")
    def test_calls_annotator_run(self, mock_run):
        """Test that annotator_cli calls the annotator run function."""
        # Arrange
        from geoparser.cli.annotator import annotator_cli

        # Act
        annotator_cli()

        # Assert
        mock_run.assert_called_once()

    @patch("geoparser.annotator.app.run")
    def test_defaults_bind_to_localhost(self, mock_run):
        """By default the annotator is local-only and opens a browser."""
        # Arrange
        from geoparser.cli.annotator import annotator_cli

        # Act
        annotator_cli()

        # Assert
        mock_run.assert_called_once_with(
            use_reloader=False, host="127.0.0.1", port=5000, open_browser=True
        )
