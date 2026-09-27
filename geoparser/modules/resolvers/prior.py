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
    DEFAULT_WEIGHT = 0.3

    def __init__(
        self,
        *args,
        population_weight: float = DEFAULT_WEIGHT,
        inflection_fallback: bool = False,
        **kwargs,
    ):
        """
        Initialize the resolver.

        Args:
            *args: Positional arguments accepted by SentenceTransformerResolver
            **kwargs: Keyword arguments accepted by SentenceTransformerResolver
            population_weight: How much the population prior counts
            inflection_fallback: Retry exact misses with trimmed names
        """
        super().__init__(
            *args,
            population_weight=population_weight,
            inflection_fallback=inflection_fallback,
            **kwargs,
        )
        self.population_weight = population_weight
        self.inflection_fallback = inflection_fallback

    def _search_candidates(
        self, name: str, method: str, tiers: int, limit: int = 10000
    ) -> tuple[Feature, ...]:
        """Search as the parent does, retrying an exact miss with trimmed names."""
        search = super()._search_candidates
        found = search(name, method, tiers, limit=limit)
        if found or not self._falls_back(method):
            return found
        return next(
            (
                variant_found
                for variant in inflection_variants(name)
                if (variant_found := search(variant, method, tiers, limit=limit))
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
        similarities: list[float],
    ) -> tuple[str, str] | None:
        """Pick the candidate with the best similarity plus population prior."""
        scores = combined_scores(
            similarities,
            [(candidate.data or {}).get("population") for candidate in candidate_list],
            self.population_weight,
        )
        best = max(range(len(scores)), key=lambda index: scores[index])
        if similarities[best] < min_similarity:
            return None
        return self.gazetteer_name, candidate_list[best].identifier
