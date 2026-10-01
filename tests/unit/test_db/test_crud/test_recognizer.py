"""
Unit tests for geoparser/db/crud/recognizer.py

Tests the RecognizerRepository class with custom query methods.
"""

import pytest
from sqlmodel import Session

from geoparser.db.crud import RecognizerRepository


@pytest.mark.unit
class TestRecognizerRepositoryGetByNameAndConfig:
    """Test the get_by_name_and_config method of RecognizerRepository."""

    def test_returns_recognizer_for_matching_name_and_config(
        self, test_session: Session, recognizer_factory
    ):
        """Test that get_by_name_and_config returns recognizer when both match."""
        # Arrange
        config = {"model": "en_core_web_sm", "threshold": 0.5}
        recognizer = recognizer_factory(name="SpacyRecognizer", config=config)

        # Act
        found_recognizer = RecognizerRepository.get_by_name_and_config(
            test_session, "SpacyRecognizer", config
        )

        # Assert
        assert found_recognizer is not None
        assert found_recognizer.id == recognizer.id
        assert found_recognizer.name == "SpacyRecognizer"
        assert found_recognizer.config == config

    def test_returns_none_for_non_matching_name(
        self, test_session: Session, recognizer_factory
    ):
        """Test that get_by_name_and_config returns None when name doesn't match."""
        # Arrange
        config = {"model": "en_core_web_sm"}
        recognizer_factory(name="SpacyRecognizer", config=config)

        # Act
        found_recognizer = RecognizerRepository.get_by_name_and_config(
            test_session, "ManualRecognizer", config
        )

        # Assert
        assert found_recognizer is None

    def test_returns_none_for_non_matching_config(
        self, test_session: Session, recognizer_factory
    ):
        """Test that get_by_name_and_config returns None when config doesn't match."""
        # Arrange
        config1 = {"model": "en_core_web_sm"}
        config2 = {"model": "en_core_web_lg"}
        recognizer_factory(name="SpacyRecognizer", config=config1)

        # Act
        found_recognizer = RecognizerRepository.get_by_name_and_config(
            test_session, "SpacyRecognizer", config2
        )

        # Assert
        assert found_recognizer is None

    @pytest.mark.parametrize(
        ("selected_config", "other_config"),
        [
            ({"model": "en_core_web_sm"}, {"model": "en_core_web_lg"}),
            ({"model": "en_core_web_lg"}, {"model": "en_core_web_sm"}),
        ],
    )
    def test_distinguishes_between_same_name_different_configs(
        self, test_session: Session, recognizer_factory, selected_config, other_config
    ):
        """The selected config still matches when another config shares its name."""
        selected = recognizer_factory(name="SpacyRecognizer", config=selected_config)
        recognizer_factory(name="SpacyRecognizer", config=other_config)

        found = RecognizerRepository.get_by_name_and_config(
            test_session, "SpacyRecognizer", selected_config
        )

        assert found is not None
        assert found.id == selected.id
