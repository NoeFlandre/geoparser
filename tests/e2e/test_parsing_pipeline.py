"""
End-to-end tests for parsing pipeline.

Tests complete end-to-end parsing workflows using real recognizers and resolvers.
Basic API functionality is covered in integration tests.
"""

import pytest

from geoparser.geoparser import Geoparser
from geoparser.modules import SentenceTransformerResolver, SpacyRecognizer
from geoparser.project import Project


def _all_toponyms(documents):
    """Collect the predicted toponyms returned for a document batch."""
    return [toponym for document in documents for toponym in document.toponyms]


def _resolved_toponyms(documents):
    """Collect predicted toponyms that resolved to a gazetteer feature."""
    return [toponym for toponym in _all_toponyms(documents) if toponym.location]


@pytest.mark.e2e
class TestCompleteParsingPipeline:
    """End-to-end tests for complete parsing workflows with real models."""

    def test_full_pipeline_with_real_models(
        self,
        real_spacy_recognizer,
        real_sentencetransformer_resolver,
        andorra_gazetteer,
    ):
        """Test complete pipeline using real spaCy recognizer and SentenceTransformer resolver."""
        # Arrange
        texts = [
            "Andorra la Vella is the capital of Andorra.",
            "The parish of Escaldes-Engordany is nearby.",
        ]

        # Act - Use Geoparser stateless API with real models
        geoparser = Geoparser(
            recognizer=real_spacy_recognizer,
            resolver=real_sentencetransformer_resolver,
        )
        documents = geoparser.parse(texts, save=False)

        # Assert
        assert len(documents) == 2
        assert _all_toponyms([documents[0]])
        assert _resolved_toponyms(documents)

    def test_project_workflow_with_real_models(
        self,
        real_spacy_recognizer,
        real_sentencetransformer_resolver,
        andorra_gazetteer,
    ):
        """Test Project-based workflow with real recognizers and resolvers."""
        # Arrange
        project = Project("e2e_real_models_test")
        texts = [
            "Encamp is a beautiful parish in Andorra.",
            "Sant Julia de Loria is in the south.",
        ]

        # Act
        project.create_documents(texts)
        project.run_recognizer(real_spacy_recognizer)
        project.run_resolver(real_sentencetransformer_resolver)

        documents = project.get_documents()

        # Assert
        assert len(documents) == 2
        assert _all_toponyms(documents)
        assert _resolved_toponyms(documents)

        # Cleanup
        project.delete()

    def test_multiple_recognizers_comparison(
        self, real_spacy_recognizer, andorra_gazetteer
    ):
        """Test workflow comparing multiple recognizer configurations on same data."""
        # Arrange
        project = Project("recognizer_comparison_test")
        # Use well-known location names that spaCy will definitely recognize
        texts = [
            "Paris is the capital of France.",
            "The city of London is a major hub.",
        ]

        # Act - Run same data through two different entity type configurations
        recognizer1 = SpacyRecognizer(
            model_name="en_core_web_sm", entity_types=["GPE", "LOC"]
        )
        recognizer2 = SpacyRecognizer(
            model_name="en_core_web_sm", entity_types=["GPE", "LOC", "FAC"]
        )

        project.create_documents(texts)
        project.run_recognizer(recognizer1, tag="config1")
        project.run_recognizer(recognizer2, tag="config2")

        # Get results for each recognizer using tags
        docs_rec1 = project.get_documents(tag="config1")
        docs_rec2 = project.get_documents(tag="config2")

        # Assert - Both should produce results (may differ)
        assert len(docs_rec1) == 2
        assert len(docs_rec2) == 2
        assert _all_toponyms(docs_rec1) or _all_toponyms(docs_rec2)

        # Cleanup
        project.delete()

    def test_multiple_resolvers_comparison(
        self, real_spacy_recognizer, andorra_gazetteer
    ):
        """Test workflow comparing multiple resolver configurations on same data."""
        # Arrange
        project = Project("resolver_comparison_test")
        texts = [
            "Paris is the capital of France.",
            "London is a major city.",
        ]

        project.create_documents(texts)

        # Run recognizer on two different tags for comparison
        project.run_recognizer(real_spacy_recognizer, tag="config1")
        project.run_recognizer(real_spacy_recognizer, tag="config2")

        # Define attribute map for andorranames gazetteer
        andorra_attribute_map = {
            "name": "name",
            "type": "feature_name",
            "level1": "country_name",
            "level2": "admin1_name",
            "level3": "admin2_name",
        }

        # Act - Run different resolver configurations with different parameters
        # Different similarity thresholds and maximum tiers to expand through
        resolver1 = SentenceTransformerResolver(
            gazetteer_name="andorranames",
            model_name="dguzh/geo-all-MiniLM-L6-v2",
            min_similarity=0.6,
            max_tiers=3,
            attribute_map=andorra_attribute_map,
        )
        resolver2 = SentenceTransformerResolver(
            gazetteer_name="andorranames",
            model_name="dguzh/geo-all-MiniLM-L6-v2",
            min_similarity=0.5,
            max_tiers=5,
            attribute_map=andorra_attribute_map,
        )

        project.run_resolver(resolver1, tag="config1")
        project.run_resolver(resolver2, tag="config2")

        # Get results for each resolver using tags
        docs_res1 = project.get_documents(tag="config1")
        docs_res2 = project.get_documents(tag="config2")

        # Assert - Both should produce results
        assert len(docs_res1) == 2
        assert len(docs_res2) == 2

        # Both resolvers should have attempted resolution
        # (may or may not succeed depending on gazetteer content)
        assert _all_toponyms(docs_res1)
        assert _all_toponyms(docs_res2)

        # Cleanup
        project.delete()

    def test_end_to_end_with_context_switching(self, monkeypatch):
        """Keep exact spans and resolver results separate across tagged contexts."""
        from types import SimpleNamespace

        from geoparser.gazetteer import gazetteer as gazetteer_module
        from geoparser.modules.recognizers.manual import ManualRecognizer
        from geoparser.modules.resolvers.manual import ManualResolver
        from geoparser.services import resolution as resolution_service_module

        class ManualGazetteer:
            def __init__(self, name: str) -> None:
                self.name = name

            def find(self, identifier: str) -> SimpleNamespace:
                return SimpleNamespace(gazetteer_name=self.name, identifier=identifier)

        monkeypatch.setattr(
            gazetteer_module, "get_gazetteer", lambda name: ManualGazetteer(name)
        )
        monkeypatch.setattr(resolution_service_module, "Gazetteer", ManualGazetteer)

        project = Project("context_switching_test")
        texts = ["Paris is a beautiful city in France."]
        paris_span = (0, 5)

        try:
            project.create_documents(texts)
            project.run_recognizer(
                ManualRecognizer(
                    label="broad-fixture", texts=texts, references=[[paris_span]]
                ),
                tag="broad",
            )
            project.run_recognizer(
                ManualRecognizer(
                    label="narrow-fixture", texts=texts, references=[[paris_span]]
                ),
                tag="narrow",
            )
            project.run_resolver(
                ManualResolver(
                    label="broad-fixture",
                    texts=texts,
                    references=[[paris_span]],
                    referents=[[("manual-gazetteer", "fixture-feature-42")]],
                ),
                tag="broad",
            )

            resolved_documents = project.get_documents(tag="broad")
            unresolved_documents = project.get_documents(tag="narrow")

            assert len(resolved_documents) == 1
            assert len(unresolved_documents) == 1
            (resolved,) = resolved_documents[0].toponyms
            (unresolved,) = unresolved_documents[0].toponyms
            assert (resolved.start, resolved.end, resolved.text) == (0, 5, "Paris")
            resolved_location = resolved.location
            assert resolved_location is not None
            assert (
                resolved_location.gazetteer_name,
                resolved_location.identifier,
            ) == (
                "manual-gazetteer",
                "fixture-feature-42",
            )
            assert (unresolved.start, unresolved.end, unresolved.text) == (
                0,
                5,
                "Paris",
            )
            assert unresolved.location is None
        finally:
            project.delete()

    def test_batch_processing_with_real_models(
        self,
        real_spacy_recognizer,
        real_sentencetransformer_resolver,
        andorra_gazetteer,
    ):
        """Test processing larger batch of documents with real models."""
        # Arrange
        project = Project("batch_processing_test")

        # Create varied texts with clear location mentions
        texts = [
            "Paris is the capital of France.",
            "London is in England.",
            "Berlin is the German capital.",
            "Rome is in Italy.",
            "Madrid is in Spain.",
            "Lisbon is the capital of Portugal.",
            "Athens is in Greece.",
        ]

        # Act
        project.create_documents(texts)
        project.run_recognizer(real_spacy_recognizer)
        project.run_resolver(real_sentencetransformer_resolver)

        documents = project.get_documents()

        # Assert
        assert len(documents) == 7

        # Should recognize multiple locations across documents
        total_toponyms = sum(len(doc.toponyms) for doc in documents)
        # With GPE and LOC entity types, we should find many locations
        assert total_toponyms > 0

        # Cleanup
        project.delete()
