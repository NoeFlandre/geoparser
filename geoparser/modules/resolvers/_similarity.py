"""Cosine-similarity scoring for SentenceTransformerResolver."""

import itertools
import typing as t

import torch

if t.TYPE_CHECKING:
    from geoparser.gazetteer.feature import Feature


class SimilarityMixin:
    """Score candidates against reference contexts from cached embeddings."""

    context_embeddings: dict[str, torch.Tensor]
    candidate_embeddings: dict[int, torch.Tensor]

    def _calculate_similarities(
        self,
        context_embedding: torch.Tensor,
        candidate_embeddings: list[torch.Tensor],
    ) -> list[float]:
        """
        Calculate cosine similarities between context and candidate embeddings.

        Args:
            context_embedding: Embedding tensor for the reference context
            candidate_embeddings: List of embedding tensors for candidates

        Returns:
            List of similarity scores
        """
        if not candidate_embeddings:
            return []

        # Stack candidate embeddings
        candidate_tensor = torch.stack(candidate_embeddings)

        # Calculate cosine similarities
        # pragma: no mutate start - dim=1 is also torch's default, so a
        # mutant that drops it computes exactly the same similarities.
        similarities = torch.nn.functional.cosine_similarity(
            context_embedding.unsqueeze(0), candidate_tensor, dim=1
        )
        # pragma: no mutate end

        return similarities.tolist()

    def _calculate_similarity_batches(
        self,
        contexts: list[str],
        candidate_lists: list[list["Feature"]],
    ) -> list[list[float]]:
        """Score every candidate list with one flattened cosine operation."""
        scores: list[list[float]] = [[] for _ in candidate_lists]
        pending = self._non_empty(contexts, candidate_lists)
        if not pending:
            return scores

        lengths = [len(candidate_list) for _, _, candidate_list in pending]
        similarities = self._flat_similarities(pending, lengths)
        offsets = [0, *itertools.accumulate(lengths)]
        for (index, _, _), start, end in zip(
            pending, offsets, offsets[1:], strict=False
        ):
            scores[index] = similarities[start:end]
        return scores

    @staticmethod
    def _non_empty(
        contexts: list[str], candidate_lists: list[list["Feature"]]
    ) -> list[tuple[int, str, list["Feature"]]]:
        """Return (index, context, candidates) for every non-empty list."""
        return [
            (index, context, candidate_list)
            for index, (context, candidate_list) in enumerate(
                zip(contexts, candidate_lists, strict=True)
            )
            if candidate_list
        ]

    def _flat_similarities(
        self,
        pending: list[tuple[int, str, list["Feature"]]],
        lengths: list[int],
    ) -> list[float]:
        """
        Score every (context, candidate) pair of a batch in one cosine call.

        Args:
            pending: (index, context, candidates) for each non-empty list
            lengths: How many candidates each entry of ``pending`` has

        Returns:
            One similarity per candidate, in the order of ``pending``
        """
        context_tensor = torch.stack(
            [self.context_embeddings[context] for _, context, _ in pending]
        )
        candidate_tensor = torch.cat(
            [
                torch.stack(
                    [self.candidate_embeddings[candidate.id] for candidate in items]
                )
                for _, _, items in pending
            ]
        )
        repeated_contexts = torch.repeat_interleave(
            context_tensor,
            torch.tensor(lengths, device=context_tensor.device),
            dim=0,
        )
        return torch.nn.functional.cosine_similarity(
            repeated_contexts, candidate_tensor, dim=1
        ).tolist()
