"""Scoring semantics, truncation, batching and malformed-output rejection.

Expected values are hand-counted from the documented templates and from the
pinned Jina code. Nothing here loads a checkpoint or runs inference.
"""

import math

import numpy as np
import pytest

from scripts.reranker_isolation.scoring import (
    JINA_BLOCK_SIZE,
    JINA_MAX_DOCUMENT_TOKENS,
    JINA_MAX_QUERY_TOKENS,
    QWEN_INSTRUCTION,
    QWEN_MAX_LENGTH,
    QWEN_PREFIX,
    QWEN_SUFFIX,
    MalformedOutputError,
    assemble_qwen_ids,
    batched,
    checked_scores,
    jina_blocks,
    listwise_scores,
    qwen_pair_text,
    qwen_token_budget,
    qwen_yes_probability,
    score_listwise,
    score_pointwise,
    truncate_tokens,
)

NAN = float("nan")
INF = float("inf")


# --- Qwen3 template and P(yes) -------------------------------------------------


def test_qwen_pair_text_follows_the_documented_template() -> None:
    text = qwen_pair_text("Paris", "City in France", instruction="Find it")

    assert text == "<Instruct>: Find it\n<Query>: Paris\n<Document>: City in France"


def test_qwen_prefix_suffix_and_instruction_are_the_documented_strings() -> None:
    assert QWEN_PREFIX == (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer can "
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    assert QWEN_SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert QWEN_INSTRUCTION == (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    assert QWEN_MAX_LENGTH == 8192


def test_qwen_yes_probability_hand_computed_values() -> None:
    assert qwen_yes_probability(0.0, 0.0) == pytest.approx(0.5)
    assert qwen_yes_probability(0.0, math.log(3)) == pytest.approx(0.75)
    assert qwen_yes_probability(math.log(3), 0.0) == pytest.approx(0.25)


def test_higher_yes_logit_gives_a_higher_score() -> None:
    assert qwen_yes_probability(0.2, 1.0) > qwen_yes_probability(0.2, 0.5)


def test_higher_no_logit_gives_a_lower_score() -> None:
    assert qwen_yes_probability(1.0, 0.2) < qwen_yes_probability(0.5, 0.2)


def test_qwen_yes_probability_is_stable_for_extreme_logits() -> None:
    assert qwen_yes_probability(1000.0, -1000.0) == 0.0
    assert qwen_yes_probability(-1000.0, 1000.0) == 1.0


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_qwen_yes_probability_rejects_non_finite_logits(bad: float) -> None:
    with pytest.raises(MalformedOutputError):
        qwen_yes_probability(bad, 0.0)
    with pytest.raises(MalformedOutputError):
        qwen_yes_probability(0.0, bad)


# --- truncation -------------------------------------------------------------------


def test_qwen_token_budget_subtracts_prefix_and_suffix() -> None:
    assert qwen_token_budget(prefix_tokens=3, suffix_tokens=4) == 8192 - 7


def test_qwen_budget_must_leave_room_for_the_document() -> None:
    with pytest.raises(ValueError, match="no room"):
        qwen_token_budget(prefix_tokens=8190, suffix_tokens=2)


def test_assembled_qwen_ids_keep_prefix_and_suffix_and_truncate_the_pair() -> None:
    ids, truncated = assemble_qwen_ids(
        prefix=[1, 2], pair=[3, 4, 5, 6], suffix=[9], max_length=6
    )

    assert ids == [1, 2, 3, 4, 5, 9]
    assert truncated is True


def test_assembled_qwen_ids_leave_a_short_pair_whole() -> None:
    ids, truncated = assemble_qwen_ids(
        prefix=[1], pair=[3, 4], suffix=[9], max_length=10
    )

    assert ids == [1, 3, 4, 9]
    assert truncated is False


def test_truncate_tokens_reports_whether_anything_was_cut() -> None:
    assert truncate_tokens([1, 2, 3, 4], 2) == ([1, 2], True)
    assert truncate_tokens([1, 2], 5) == ([1, 2], False)
    assert truncate_tokens([1, 2], 2) == ([1, 2], False)


def test_truncation_budget_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="negative"):
        truncate_tokens([1], -1)


def test_jina_limits_match_the_pinned_modeling_code() -> None:
    assert JINA_MAX_QUERY_TOKENS == 1024
    assert JINA_MAX_DOCUMENT_TOKENS == 8192
    assert JINA_BLOCK_SIZE == 125


# --- batching -----------------------------------------------------------------------


def test_batched_keeps_order_and_gives_a_short_final_batch() -> None:
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_batched_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(batched([1], 0))


def test_score_pointwise_batches_and_keeps_the_prompt_order() -> None:
    calls: list[list[str]] = []

    def logits(batch: list[str]) -> list[tuple[float, float]]:
        calls.append(list(batch))
        return [(0.0, float(len(text))) for text in batch]

    scores = score_pointwise(logits, ["a", "bb", "ccc"], batch_size=2)

    assert calls == [["a", "bb"], ["ccc"]]
    # With no=0 and yes=len, P(yes) = 1 / (1 + exp(-len)), a sigmoid of len.
    expected = [1 / (1 + math.exp(-length)) for length in (1, 2, 3)]
    assert scores == pytest.approx(expected)


def test_score_pointwise_rejects_a_batch_with_too_few_pairs() -> None:
    def short(batch: list[str]) -> list[tuple[float, float]]:
        return [(0.0, 0.0)]

    with pytest.raises(MalformedOutputError, match="expected 2"):
        score_pointwise(short, ["a", "b"], batch_size=2)


@pytest.mark.parametrize("pair", [(0.0,), (NAN, 0.0), (0.0, INF), "yes"])
def test_score_pointwise_rejects_malformed_logit_pairs(pair: object) -> None:
    def malformed(batch: list[str]) -> list[object]:
        return [pair for _ in batch]

    with pytest.raises(MalformedOutputError):
        score_pointwise(malformed, ["a"], batch_size=1)


# --- listwise (Jina) ------------------------------------------------------------------


def test_listwise_scores_are_aligned_to_shortlist_positions() -> None:
    results = [
        {"index": 2, "relevance_score": 0.1},
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.5},
    ]

    assert listwise_scores(results, 3) == [0.9, 0.5, 0.1]


def test_listwise_scores_accept_the_numpy_scalars_the_model_returns() -> None:
    results = [
        {"index": np.int64(1), "relevance_score": np.float32(0.25)},
        {"index": np.int64(0), "relevance_score": np.float32(-0.5)},
    ]

    assert listwise_scores(results, 2) == pytest.approx([-0.5, 0.25])


@pytest.mark.parametrize(
    "results",
    [
        [],
        [{"index": 0, "relevance_score": 0.5}],
        [
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.4},
            {"index": 1, "relevance_score": 0.3},
        ],
    ],
    ids=["empty", "too-few", "too-many"],
)
def test_listwise_rejects_the_wrong_number_of_results(
    results: list[dict[str, object]],
) -> None:
    with pytest.raises(MalformedOutputError, match="expected 2 results"):
        listwise_scores(results, 2)


def test_a_well_formed_pair_of_results_is_accepted() -> None:
    results = [
        {"index": 0, "relevance_score": 0.5},
        {"index": 1, "relevance_score": 0.4},
    ]

    assert listwise_scores(results, 2) == [0.5, 0.4]


@pytest.mark.parametrize(
    "bad_result",
    [
        {"index": 0, "relevance_score": 0.5},  # duplicate of the other result
        {"index": 2, "relevance_score": 0.4},  # past the end of the shortlist
        {"index": -1, "relevance_score": 0.4},  # negative
        {"index": True, "relevance_score": 0.4},  # bool is not an index
        {"index": 1.0, "relevance_score": 0.4},  # float is not an index
        {"index": "1", "relevance_score": 0.4},  # string is not an index
        {"relevance_score": 0.4},  # missing index
        {"index": 1},  # missing score
        {"index": 1, "relevance_score": NAN},
        {"index": 1, "relevance_score": INF},
        {"index": 1, "relevance_score": "0.4"},  # string score
        {"index": 1, "relevance_score": True},  # bool score
        "0.4",  # not a mapping at all
    ],
)
def test_listwise_rejects_malformed_results(bad_result: object) -> None:
    results = [{"index": 0, "relevance_score": 0.5}, bad_result]

    with pytest.raises(MalformedOutputError):
        listwise_scores(results, 2)


def test_score_listwise_calls_the_reranker_once_with_the_whole_shortlist() -> None:
    calls: list[list[str]] = []

    def rerank(documents: list[str]) -> list[dict[str, object]]:
        calls.append(list(documents))
        return [
            {"index": position, "relevance_score": 1.0 - position / 10}
            for position in range(len(documents))
        ]

    scores = score_listwise(rerank, ["d0", "d1", "d2"])

    assert calls == [["d0", "d1", "d2"]]
    assert scores == pytest.approx([1.0, 0.9, 0.8])


def test_score_listwise_refuses_an_empty_shortlist_without_calling() -> None:
    calls: list[list[str]] = []

    def rerank(documents: list[str]) -> list[dict[str, object]]:
        calls.append(list(documents))
        return []

    with pytest.raises(ValueError, match="no documents"):
        score_listwise(rerank, [])
    assert calls == []


def test_checked_scores_rejects_length_mismatch_and_non_finite_values() -> None:
    assert checked_scores([0.25, -0.5], 2) == [0.25, -0.5]
    with pytest.raises(MalformedOutputError, match="expected 2"):
        checked_scores([0.1], 2)
    with pytest.raises(MalformedOutputError, match="not a finite"):
        checked_scores([NAN, 0.1], 2)


# --- Jina internal blocking -----------------------------------------------------------


def test_jina_blocks_keep_a_small_shortlist_in_one_call() -> None:
    # The pinned tokenizer reports model_max_length 131072.
    blocks = jina_blocks([10, 10, 10], query_length=2, model_max_length=131072)

    assert blocks == [[0, 1, 2]]


def test_jina_blocks_flush_every_document_when_the_model_length_is_tiny() -> None:
    # Remaining capacity is compared with the 8192-token document limit, so a
    # model length of 100 always falls below it and every document is flushed.
    blocks = jina_blocks([10, 10, 10], query_length=2, model_max_length=100)

    assert blocks == [[0], [1], [2]]


def test_jina_blocks_flush_when_the_capacity_reaches_the_document_limit() -> None:
    # After the first document capacity is 10000 - 2000 = 8000, which is at most
    # 8192, so each of the first two documents forms its own block.
    blocks = jina_blocks([2000, 2000, 10], query_length=0, model_max_length=10000)

    assert blocks == [[0], [1], [2]]


def test_jina_blocks_flush_at_the_block_size() -> None:
    blocks = jina_blocks([1] * 130, query_length=0, model_max_length=131072)

    assert blocks == [list(range(125)), list(range(125, 130))]


def test_jina_blocks_of_an_empty_list_are_empty() -> None:
    assert jina_blocks([], query_length=0, model_max_length=131072) == []
