"""
A sentence-transformer resolver tuned for multilingual text.

It keeps the upstream geocoding encoder and tiered gazetteer search, and
changes two things (see :mod:`~geoparser.modules.resolvers.ranking`): an exact
search that finds nothing is retried with inflection-trimmed names before the
search is widened, and candidates are ranked by similarity plus a small
population prior. Both are cheap; neither loads another model.
"""

from __future__ import annotations

import typing as t

from geoparser.modules.resolvers.ranking import combined_scores, inflection_variants
from geoparser.modules.resolvers.sentencetransformer import SentenceTransformerResolver

if t.TYPE_CHECKING:
    from geoparser.gazetteer.feature import Feature


class PriorResolver(SentenceTransformerResolver):
    """The upstream resolver with inflection fallback and a population prior."""

    NAME = "PriorResolver"

    # Chosen so a million-person city gains about 0.06 of similarity: enough
    # to decide a near-tie between homonyms, too little to overturn context.
    DEFAULT_WEIGHT = 0.1

    def __init__(  # noqa: PLR0913, PLR0917 - public API; make keyword-only in the next major release
        self,
        model_name: str = "dguzh/geo-all-MiniLM-L6-v2",
        gazetteer_name: str = "geonames",
        min_similarity: float = 0.6,
        max_tiers: int = 3,
        attribute_map: dict | None = None,
        population_weight: float = DEFAULT_WEIGHT,
        inflection_fallback: bool = True,  # noqa: FBT001, FBT002 - positional bool kept for API compatibility; make keyword-only in the next major release
    ):
        """
        Initialize the resolver.

        Args:
            model_name: HuggingFace model name for SentenceTransformer
            gazetteer_name: Name of the gazetteer to search
            min_similarity: Raw similarity a candidate must reach; the prior
                ranks candidates but cannot lift one over this threshold
            max_tiers: Maximum number of tiers to expand through search methods
            attribute_map: Optional custom attribute mapping for the gazetteer
            population_weight: How much the population prior counts
            inflection_fallback: Retry exact misses with trimmed names
        """
        super().__init__(
            model_name=model_name,
            gazetteer_name=gazetteer_name,
            min_similarity=min_similarity,
            max_tiers=max_tiers,
            attribute_map=attribute_map,
            population_weight=population_weight,
            inflection_fallback=inflection_fallback,
        )
        self.population_weight = population_weight
        self.inflection_fallback = inflection_fallback

    def _search_candidates(
        self, name: str, method: str, tiers: int
    ) -> tuple[Feature, ...]:
        """Search as the parent does, retrying an exact miss with trimmed names."""
        search = super()._search_candidates
        found = search(name, method, tiers)
        if found or not self._falls_back(method):
            return found
        return next(
            (
                variant_found
                for variant in inflection_variants(name)
                if (variant_found := search(variant, method, tiers))
            ),
            found,
        )

    def _falls_back(self, method: str) -> bool:
        """Whether a miss by this search method is retried with trimmed names."""
        return self.inflection_fallback and method == "exact"

    def _best_referent(
        self,
        context: str,
        candidate_list: list[Feature],
        min_similarity: float,
        similarities: list[float] | None = None,
    ) -> tuple[str, str] | None:
        """Pick the candidate with the best similarity plus population prior."""
        if similarities is None:
            similarities = self._context_similarities(context, candidate_list)
        scores = combined_scores(
            similarities,
            [(candidate.data or {}).get("population") for candidate in candidate_list],
            self.population_weight,
        )
        best = max(range(len(scores)), key=lambda index: scores[index])
        if similarities[best] < min_similarity:
            return None
        return self.gazetteer_name, candidate_list[best].identifier

    def _context_similarities(
        self, context: str, candidate_list: list[Feature]
    ) -> list[float]:
        """Score each candidate against the context, from cached embeddings."""
        return self._calculate_similarities(
            self.context_embeddings[context],
            [self.candidate_embeddings[candidate.id] for candidate in candidate_list],
        )
