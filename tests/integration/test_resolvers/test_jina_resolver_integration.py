"""
Integration tests for geoparser/modules/resolvers/jina.py

These load the real Jina embedding model and reranker and resolve against a
real gazetteer artifact. The unit tests pin the shortlisting and the handover
to the reranker with mocks; what they cannot check is that the prompt names
this resolver uses are ones the checkpoint actually defines, or that the
reranker's return shape is the one the code unpacks. Both fail silently under
a Mock and loudly here.

The checkpoints are large downloads, so these are opt-in: set
``GEOPARSER_TEST_REMOTE_MODELS=1`` to run them.
"""

import os
from collections.abc import Iterator

import pytest
from huggingface_hub import snapshot_download

from geoparser.modules.resolvers.jina import JinaResolver

EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
EMBEDDING_MODEL_REVISION = "dd76d535f5447ca3897a9c893fb1e612ead98192"
RERANKER_MODEL_ID = "jinaai/jina-reranker-v3.5"
RERANKER_MODEL_REVISION = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"

pytestmark = pytest.mark.skipif(
    not os.getenv("GEOPARSER_TEST_REMOTE_MODELS"),
    reason="Set GEOPARSER_TEST_REMOTE_MODELS=1 to download and run the Jina models.",
)

ANDORRA_ATTRIBUTE_MAP = {
    "name": "name",
    "type": "feature_name",
    "level1": "country_name",
    "level2": "admin1_name",
    "level3": "admin2_name",
}


@pytest.fixture(scope="module")
def resolver(session_geoparser_data_dir) -> Iterator[JinaResolver]:
    """A real Jina resolver pointed at the Andorra test gazetteer."""
    old_data_dir = os.environ.get("GEOPARSER_DATA_DIR")
    old_gazetteers_dir = os.environ.get("GEOPARSER_GAZETTEERS_DIR")
    try:
        os.environ["GEOPARSER_DATA_DIR"] = str(session_geoparser_data_dir)
        os.environ.pop("GEOPARSER_GAZETTEERS_DIR", None)
        embedding_path = snapshot_download(
            repo_id=EMBEDDING_MODEL_ID,
            revision=EMBEDDING_MODEL_REVISION,
        )
        reranker_path = snapshot_download(
            repo_id=RERANKER_MODEL_ID,
            revision=RERANKER_MODEL_REVISION,
        )
        yield JinaResolver(
            model_name=embedding_path,
            reranker_name=reranker_path,
            gazetteer_name="andorranames",
            min_similarity=0.0,
            max_tiers=2,
            attribute_map=ANDORRA_ATTRIBUTE_MAP,
        )
    finally:
        if old_data_dir is None:
            os.environ.pop("GEOPARSER_DATA_DIR", None)
        else:
            os.environ["GEOPARSER_DATA_DIR"] = old_data_dir
        if old_gazetteers_dir is None:
            os.environ.pop("GEOPARSER_GAZETTEERS_DIR", None)
        else:
            os.environ["GEOPARSER_GAZETTEERS_DIR"] = old_gazetteers_dir


@pytest.mark.integration
class TestJinaResolverIntegration:
    """The real models, against a real gazetteer."""

    def test_resolves_a_place_name_to_a_gazetteer_entry(self, resolver):
        """A named Andorran place is linked to an identifier."""
        # Arrange
        text = "Andorra la Vella is the capital of Andorra."

        # Act
        ((referent,),) = resolver.predict([text], [[(0, 16)]])

        # Assert
        assert referent is not None
        gazetteer_name, identifier = referent
        assert gazetteer_name == "andorranames"
        assert resolver.gazetteer.find(identifier) is not None

    def test_returns_one_slot_per_reference(self, resolver):
        """The result mirrors the reference structure it was given."""
        # Arrange
        text = "Encamp and Canillo are parishes of Andorra."

        # Act
        results = resolver.predict([text], [[(0, 6), (11, 18)]])

        # Assert
        assert len(results) == 1
        assert len(results[0]) == 2

    def test_the_embedding_threshold_can_reject_every_candidate(
        self, resolver, monkeypatch
    ):
        """
        A threshold no candidate reaches leaves the reference unresolved.

        This is what stops the reranker from being handed a shortlist of
        candidates the embedding stage had no confidence in at all.
        """
        # Arrange
        monkeypatch.setattr(resolver, "min_similarity", 1.1)

        # Act
        ((referent,),) = resolver.predict(["Encamp is a parish."], [[(0, 6)]])

        # Assert
        assert referent is None

    def test_the_reranker_returns_an_index_into_the_shortlist(self, resolver):
        """
        The reranker's output shape is the one the code unpacks.

        `rerank` returns dictionaries carrying an `index` into the documents
        it was given; reading the wrong key, or assuming the results come back
        in input order, would pick an arbitrary candidate rather than fail.
        """
        # Arrange
        candidates = resolver.gazetteer.search("Encamp", method="exact", limit=5)
        assert candidates, "the Andorra gazetteer should contain Encamp"
        descriptions = list(map(resolver._generate_description, candidates))

        # Act
        ranking = resolver.reranker.rerank("Encamp, a parish", descriptions, top_n=1)

        # Assert
        assert len(ranking) == 1
        assert 0 <= ranking[0]["index"] < len(descriptions)
        # A real number, however the model spells it: the score comes back as
        # a numpy scalar rather than a Python float.
        assert float(ranking[0]["relevance_score"]) == ranking[0]["relevance_score"]

    def test_the_two_encoding_roles_produce_different_embeddings(self, resolver):
        """
        Query and document prompts are genuinely distinct.

        If the checkpoint silently ignored the prompt names, the two roles
        would embed identically and the asymmetry this resolver is built
        around would be doing nothing.
        """
        # Arrange
        text = "Encamp"

        # Act
        as_query = resolver._encode([text], role="context")
        as_document = resolver._encode([text], role="candidate")

        # Assert
        assert as_query.shape == as_document.shape
        assert not as_query.allclose(as_document)

    def test_no_references_means_no_work(self, resolver):
        """A document with nothing to resolve returns an empty slot list."""
        # Act & Assert
        assert resolver.predict(["Nothing here."], [[]]) == [[]]
