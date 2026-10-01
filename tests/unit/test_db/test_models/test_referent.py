"""
Unit tests for geoparser/db/models/referent.py

Tests the Referent model.
"""

import uuid

import pytest
from sqlmodel import Session

from geoparser.db.models import ReferentCreate, ReferentUpdate


@pytest.mark.unit
def test_referent_factory_creates_default_parent_records(referent_factory):
    referent = referent_factory(feature_identifier="fixture-3041563")

    assert (
        referent.gazetteer_name,
        referent.feature_identifier,
        referent.reference_id is not None,
        referent.resolver_id is not None,
    ) == ("andorranames", "fixture-3041563", True, True)


@pytest.fixture
def saved_referent(test_session, reference_factory, resolver_factory):
    """Persist a referent together with its reference and resolver IDs."""
    from geoparser.db.models import Referent

    reference = reference_factory()
    resolver = resolver_factory(id="test_resolver")
    referent = Referent(
        reference_id=reference.id,
        gazetteer_name="andorranames",
        feature_identifier="3041563",
        resolver_id=resolver.id,
    )
    test_session.add(referent)
    test_session.commit()
    test_session.refresh(referent)
    return referent, {
        "reference_id": reference.id,
        "gazetteer_name": "andorranames",
        "feature_identifier": "3041563",
        "resolver_id": resolver.id,
    }


@pytest.fixture
def fully_populated_referent_update():
    """Build an update alongside the values it must retain."""
    values = {
        "id": uuid.uuid4(),
        "reference_id": uuid.uuid4(),
        "gazetteer_name": "geonames",
        "feature_identifier": "456",
        "resolver_id": "new_resolver",
    }
    return ReferentUpdate(**values), values


@pytest.fixture
def empty_referent_update():
    """Build an update with only its required identifier."""
    referent_id = uuid.uuid4()
    return ReferentUpdate(id=referent_id), referent_id


@pytest.mark.unit
class TestReferentModel:
    """Test the Referent model."""

    def test_created_referent_has_uuid(self, saved_referent):
        referent, _ = saved_referent
        assert referent.id is not None
        assert isinstance(referent.id, uuid.UUID)

    @pytest.mark.parametrize(
        "field", ("reference_id", "gazetteer_name", "feature_identifier", "resolver_id")
    )
    def test_created_referent_keeps_supplied_fields(self, saved_referent, field):
        referent, expected_values = saved_referent
        assert getattr(referent, field) == expected_values[field]

    def test_generates_uuid_automatically(
        self,
        test_session: Session,
        reference_factory,
        resolver_factory,
    ):
        """Test that Referent automatically generates a UUID for id."""
        # Arrange
        from geoparser.db.models import Referent

        reference = reference_factory()
        resolver = resolver_factory(id="test")

        referent = Referent(
            reference_id=reference.id,
            gazetteer_name="andorranames",
            feature_identifier="3041563",
            resolver_id=resolver.id,
        )

        # Act
        test_session.add(referent)
        test_session.commit()

        # Assert
        assert referent.id is not None
        assert isinstance(referent.id, uuid.UUID)

    def test_has_reference_relationship(self, test_session: Session):
        """Test that Referent has a relationship to reference."""
        # Arrange
        from geoparser.db.models import Referent

        referent = Referent(
            reference_id=uuid.uuid4(),
            gazetteer_name="andorranames",
            feature_identifier="1",
            resolver_id="test",
        )

        # Assert
        assert hasattr(referent, "reference")

    def test_has_resolver_relationship(self, test_session: Session):
        """Test that Referent has a relationship to resolver."""
        # Arrange
        from geoparser.db.models import Referent

        referent = Referent(
            reference_id=uuid.uuid4(),
            gazetteer_name="andorranames",
            feature_identifier="1",
            resolver_id="test",
        )

        # Assert
        assert hasattr(referent, "resolver")

    def test_feature_property_resolves_through_gazetteer(self, test_session: Session):
        """Test that Referent.feature looks the feature up in its gazetteer."""
        from unittest.mock import Mock, patch

        from geoparser.db.models import Referent

        referent = Referent(
            reference_id=uuid.uuid4(),
            gazetteer_name="andorranames",
            feature_identifier="3041563",
            resolver_id="test",
        )
        fake_feature = Mock()

        with patch("geoparser.gazetteer.gazetteer.Gazetteer") as mock_gazetteer:
            mock_gazetteer.return_value.find.return_value = fake_feature

            feature = referent.feature

        mock_gazetteer.assert_called_once_with("andorranames")
        mock_gazetteer.return_value.find.assert_called_once_with("3041563")
        assert feature is fake_feature


@pytest.mark.unit
class TestReferentCreate:
    """Test the ReferentCreate model."""

    def test_creates_with_required_fields(self):
        """Test that ReferentCreate can be created with required fields."""
        # Arrange
        reference_id = uuid.uuid4()
        resolver_id = "test_resolver"

        # Act
        referent_create = ReferentCreate(
            reference_id=reference_id,
            gazetteer_name="andorranames",
            feature_identifier="123",
            resolver_id=resolver_id,
        )

        # Assert
        assert referent_create.reference_id == reference_id
        assert referent_create.gazetteer_name == "andorranames"
        assert referent_create.feature_identifier == "123"
        assert referent_create.resolver_id == resolver_id


@pytest.mark.unit
class TestReferentUpdate:
    """Test the ReferentUpdate model."""

    @pytest.mark.parametrize(
        "field",
        ("id", "reference_id", "gazetteer_name", "feature_identifier", "resolver_id"),
    )
    def test_stores_each_supplied_field(self, fully_populated_referent_update, field):
        update, expected_values = fully_populated_referent_update
        assert getattr(update, field) == expected_values[field]

    @pytest.mark.parametrize(
        "field", ("reference_id", "gazetteer_name", "feature_identifier", "resolver_id")
    )
    def test_optional_fields_default_to_none(self, empty_referent_update, field):
        update, _ = empty_referent_update
        assert getattr(update, field) is None
