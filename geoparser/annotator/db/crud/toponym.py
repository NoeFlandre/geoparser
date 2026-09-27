import typing as t
import uuid

from pyproj import Transformer
from pyproj.exceptions import ProjError
from shapely.errors import GEOSException
from sqlmodel import Session as DBSession
from sqlmodel import select

from geoparser.annotator.db.crud.base import BaseRepository
from geoparser.annotator.db.models.toponym import (
    AnnotatorToponym,
    AnnotatorToponymBase,
    AnnotatorToponymCreate,
    AnnotatorToponymUpdate,
)
from geoparser.annotator.exceptions import (
    ToponymNotFoundException,
    ToponymOverlapException,
)
from geoparser.annotator.models.api import CandidatesGet
from geoparser.gazetteer.description import GAZETTEER_ATTRIBUTE_MAP, describe_feature
from geoparser.gazetteer.gazetteer import get_gazetteer

if t.TYPE_CHECKING:
    from geoparser.annotator.db.models.document import AnnotatorDocument
    from geoparser.gazetteer.feature import Feature


# _remove_duplicates only ever returns elements of new_toponyms, so it keeps
# whichever kind of toponym model it was handed.
NewToponymT = t.TypeVar("NewToponymT", bound=AnnotatorToponymBase)


class ToponymRepository(BaseRepository[AnnotatorToponym]):
    model = AnnotatorToponym
    exception_factory: t.Callable[[str, uuid.UUID], Exception] = lambda x, y: (
        ToponymNotFoundException(f"{x} with ID {y} not found.")
    )

    # Gazetteer-specific attribute mappings for location descriptions, shared
    # with SentenceTransformerResolver so the two describe features alike.
    GAZETTEER_ATTRIBUTE_MAP: t.ClassVar[dict[str, dict[str, str]]] = (
        GAZETTEER_ATTRIBUTE_MAP
    )

    # Filter attributes for each gazetteer
    GAZETTEER_FILTER_ATTRIBUTES: t.ClassVar[dict[str, list[str]]] = {
        "geonames": [
            "feature_name",
            "country_name",
            "admin1_name",
            "admin2_name",
        ],
        "swissnames3d": [
            "OBJEKTART",
            "KANTON_NAME",
            "BEZIRK_NAME",
            "GEMEINDE_NAME",
        ],
    }

    @classmethod
    def _generate_location_description(
        cls, feature: "Feature", gazetteer_name: str
    ) -> str:
        """
        Generate a lightweight textual description for a feature.

        This is a simplified version that doesn't require loading heavy ML models,
        making it fast for the annotator UI.

        Args:
            feature: Feature object
            gazetteer_name: Name of the gazetteer

        Returns:
            Location description string
        """
        location_data = feature.data
        attr_map = cls.GAZETTEER_ATTRIBUTE_MAP.get(gazetteer_name)
        if not location_data or attr_map is None:
            return feature.identifier

        return describe_feature(location_data, attr_map) or feature.identifier

    @classmethod
    def validate_overlap(
        cls,
        db: DBSession,
        toponym: AnnotatorToponymBase | AnnotatorToponymUpdate,
        document_id: uuid.UUID | str | None,
    ) -> bool:
        if not cls._has_complete_overlap_span(toponym, document_id):
            return True

        overlapping = db.exec(
            select(AnnotatorToponym).where(
                *cls._overlap_filter_args(toponym, document_id)
            )
        ).all()
        if overlapping:
            msg = f"Toponyms overlap: {overlapping} and {toponym}"
            raise ToponymOverlapException(
                msg,
            )
        return True

    @staticmethod
    def _has_complete_overlap_span(
        toponym: AnnotatorToponymBase | AnnotatorToponymUpdate,
        document_id: uuid.UUID | str | None,
    ) -> bool:
        """Return whether both the document and candidate span are known."""
        return (
            document_id is not None
            and toponym.start is not None
            and toponym.end is not None
        )

    @staticmethod
    def _overlap_filter_args(
        toponym: AnnotatorToponymBase | AnnotatorToponymUpdate,
        document_id: uuid.UUID | str | None,
    ) -> list[t.Any]:
        """Build the SQL filters while excluding an updated row by its ID."""
        start = t.cast(int, toponym.start)
        end = t.cast(int, toponym.end)
        filter_args = [
            AnnotatorToponym.document_id == document_id,
            (AnnotatorToponym.start < end) & (AnnotatorToponym.end > start),
        ]
        if hasattr(toponym, "id"):
            filter_args.append(AnnotatorToponym.id != toponym.id)
        return filter_args

    @classmethod
    def _remove_duplicates(
        cls,
        old_toponyms: t.Sequence[AnnotatorToponym | AnnotatorToponymCreate],
        new_toponyms: t.Sequence[NewToponymT],
    ) -> list[NewToponymT]:
        # only add the new toponym if there is no existing one
        toponyms = [
            new_toponym
            for new_toponym in new_toponyms
            if not cls._get_toponym(old_toponyms, new_toponym.start, new_toponym.end)
        ]
        return sorted(toponyms, key=lambda x: x.start)

    @classmethod
    def _get_wgs84_coordinates(
        cls, feature: "Feature"
    ) -> tuple[float, float] | tuple[None, None]:
        """
        Extract WGS84 (lat, lon) coordinates from a feature's geometry.

        Gazetteer artifacts declare the CRS of their geometries; coordinates
        are transformed to WGS84 when the artifact uses a different CRS.

        Args:
            feature: Feature object with geometry

        Returns:
            Tuple of (latitude, longitude) in WGS84, or (None, None) if unavailable
        """
        if not feature.geometry:
            return None, None

        try:
            # Get the centroid for point representation
            centroid = feature.geometry.centroid

            # If already in WGS84, return as-is
            if feature.crs == "EPSG:4326":
                return centroid.y, centroid.x  # lat, lon

            # Otherwise, transform to WGS84
            transformer = Transformer.from_crs(feature.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(centroid.x, centroid.y)
        except (GEOSException, ProjError):
            return None, None
        else:
            return lat, lon

    @classmethod
    def _candidate_entry(cls, candidate: "Feature", gazetteer_name: str) -> dict:
        """Convert a gazetteer feature to the candidate payload used by the UI."""
        latitude, longitude = cls._get_wgs84_coordinates(candidate)
        return {
            "loc_id": candidate.identifier,
            "description": cls._generate_location_description(
                candidate, gazetteer_name
            ),
            "attributes": candidate.data,
            "latitude": latitude,
            "longitude": longitude,
        }

    @classmethod
    def _existing_candidate_entry(
        cls,
        gazetteer: t.Any,
        gazetteer_name: str,
        loc_id: str | None,
        candidates: t.Sequence["Feature"],
    ) -> dict | None:
        """Resolve an annotated location only when search did not find it."""
        candidate_ids = {candidate.identifier for candidate in candidates}
        if not loc_id or loc_id in candidate_ids:
            return None
        existing_feature = gazetteer.find(loc_id)
        if existing_feature is None:
            return None
        return cls._candidate_entry(existing_feature, gazetteer_name)

    @classmethod
    def get_candidate_descriptions(
        cls,
        gazetteer_name: str,
        toponym: AnnotatorToponym,
        toponym_text: str,
        query_text: str,
    ) -> tuple[list[dict], bool]:
        # Initialize gazetteer
        gazetteer = get_gazetteer(gazetteer_name)

        search_text = query_text or toponym_text
        candidates = gazetteer.search(search_text, method="exact")

        candidate_descriptions = [
            cls._candidate_entry(candidate, gazetteer_name) for candidate in candidates
        ]
        existing_candidate = cls._existing_candidate_entry(
            gazetteer, gazetteer_name, toponym.loc_id, candidates
        )
        if existing_candidate is not None:
            candidate_descriptions.append(existing_candidate)
        return candidate_descriptions, existing_candidate is not None

    @classmethod
    # BaseRepository declares the widest input type (SQLModel); each repository
    # deliberately accepts its own Create/Update model. Callers always go
    # through the concrete repository, so the precise signature is worth more
    # here than strict substitutability.
    def create(  # ty: ignore[invalid-method-override]
        cls,
        db: DBSession,
        item: AnnotatorToponymCreate,
        exclude: list[str] | None = None,
        additional: dict[str, t.Any] | None = None,
    ) -> AnnotatorToponym:
        # An explicit check, not an assert: asserts vanish under ``python -O``.
        if not additional or "document_id" not in additional:
            msg = "toponym cannot be created without link to document"
            raise ValueError(msg)
        cls.validate_overlap(db, item, additional["document_id"])
        return super().create(db, item, exclude=exclude, additional=additional)

    @classmethod
    def read(cls, db: DBSession, id: uuid.UUID) -> AnnotatorToponym:
        return super().read(db, id)

    @classmethod
    def _get_toponym(
        cls,
        toponyms: t.Sequence[AnnotatorToponym | AnnotatorToponymCreate],
        start: int,
        end: int,
    ) -> AnnotatorToponym | AnnotatorToponymCreate | None:
        return next(
            (t for t in toponyms if t.start == start and t.end == end),
            None,
        )

    @classmethod
    def get_toponym(
        cls, document: "AnnotatorDocument", start: int, end: int
    ) -> AnnotatorToponym | None:
        # document.toponyms holds persisted rows, so the lookup can only yield
        # an AnnotatorToponym or None.
        found = cls._get_toponym(list(document.toponyms), start, end)
        return found if isinstance(found, AnnotatorToponym) else None

    @classmethod
    def read_all(cls, db: DBSession, **filters) -> list[AnnotatorToponym]:
        return super().read_all(db, **filters)

    @classmethod
    def get_candidates(
        cls,
        doc: "AnnotatorDocument",
        gazetteer_name: str,
        candidates_request: CandidatesGet,
    ) -> dict:
        start, end, text, query_text = cls._normalize_candidate_request(
            candidates_request
        )
        toponym = cls.get_toponym(doc, start, end)
        if not toponym:
            raise ToponymNotFoundException
        candidate_descriptions, existing_candidate_is_appended = (
            cls.get_candidate_descriptions(
                gazetteer_name,
                toponym,
                text,
                query_text,
            )
        )
        return cls._candidate_payload(
            candidate_descriptions,
            toponym,
            gazetteer_name,
            existing_candidate_is_appended=existing_candidate_is_appended,
        )

    @staticmethod
    def _normalize_candidate_request(
        request: CandidatesGet,
    ) -> tuple[int, int, str, str]:
        """Replace optional candidate-request values with their UI defaults."""
        return (
            request.start or 0,
            request.end or 0,
            request.text or "",
            request.query_text or "",
        )

    @classmethod
    def _candidate_payload(
        cls,
        candidate_descriptions: list[dict],
        toponym: AnnotatorToponym,
        gazetteer_name: str,
        *,
        existing_candidate_is_appended: bool,
    ) -> dict:
        """Build the complete response payload for candidate lookup."""
        # Get filter attributes for this gazetteer
        filter_attributes = cls.GAZETTEER_FILTER_ATTRIBUTES.get(gazetteer_name, [])
        return {
            "candidates": candidate_descriptions,
            "filter_attributes": filter_attributes,
            "existing_loc_id": toponym.loc_id,
            "existing_candidate": (
                candidate_descriptions[-1] if existing_candidate_is_appended else None
            ),
        }

    @classmethod
    def update(  # ty: ignore[invalid-method-override]
        cls,
        db: DBSession,
        item: AnnotatorToponymUpdate | AnnotatorToponym,
        document_id: uuid.UUID | str | None = None,
    ) -> AnnotatorToponym:
        cls.validate_overlap(db, item, document_id or item.document_id)
        return super().update(db, item)

    @classmethod
    def annotate_many(
        cls,
        db: DBSession,
        document: "AnnotatorDocument",
        annotation: AnnotatorToponymBase,
    ) -> list[AnnotatorToponym]:
        toponym = cls.get_toponym(document, annotation.start, annotation.end)
        if toponym is None:
            raise ToponymNotFoundException
        one_sense_per_discourse = (
            toponym.document.session.settings.one_sense_per_discourse
        )
        toponym.loc_id = annotation.loc_id if annotation.loc_id is not None else None
        cls.update(db, toponym)
        if one_sense_per_discourse:
            cls._propagate_discourse_annotation(db, document, toponym)
        db.refresh(document)
        return document.toponyms

    @staticmethod
    def _is_unannotated_repeat(
        candidate: AnnotatorToponym, selected: AnnotatorToponym
    ) -> bool:
        """Return whether a row is another unannotated mention of the selection."""
        return (
            candidate.text == selected.text
            and candidate.loc_id == ""
            and candidate is not selected
        )

    @classmethod
    def _propagate_discourse_annotation(
        cls,
        db: DBSession,
        document: "AnnotatorDocument",
        selected: AnnotatorToponym,
    ) -> None:
        """Copy a chosen location ID to other unannotated repeated mentions."""
        if not selected.loc_id:
            return
        for candidate in document.toponyms:
            if cls._is_unannotated_repeat(candidate, selected):
                candidate.loc_id = selected.loc_id
                cls.update(db, candidate)

    @classmethod
    def delete(cls, db: DBSession, id: uuid.UUID) -> AnnotatorToponym:
        return super().delete(db, id)
