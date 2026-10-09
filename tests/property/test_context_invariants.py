"""
Property tests for the context window.

The window is chosen by arithmetic over sentence costs, and the arithmetic has
to hold for any document and any budget, not only the handful of shapes the
unit tests spell out. These generate documents and budgets and assert the
properties the resolver depends on: the reference is always in the context,
the context is a contiguous run of whole sentences, it respects the budget
whenever that is possible at all, and it is as large as the budget allows.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geoparser.modules.resolvers.context import (
    Sentence,
    expand_window,
    locate_sentence,
    select_context,
)

COSTS = st.lists(st.integers(min_value=0, max_value=40), min_size=1, max_size=12)
BUDGETS = st.integers(min_value=0, max_value=120)


def _document(costs: list[int]) -> list[Sentence]:
    """A document whose sentences carry the given token costs."""
    sentences = []
    offset = 0
    for index, cost in enumerate(costs):
        text = f"s{index}" * max(cost, 1)
        sentences.append(
            Sentence(text=text, start=offset, end=offset + len(text), cost=cost)
        )
        offset += len(text) + 1
    return sentences


@st.composite
def documents_with_target(draw):
    """A document, and the index of one of its sentences."""
    costs = draw(COSTS)
    sentences = _document(costs)
    return sentences, draw(st.integers(min_value=0, max_value=len(sentences) - 1))


def _reference_take_preceding(
    costs: list[int], first: int, last: int, budget: int
) -> int:
    """Return the preceding index if adding that slice fits the budget."""
    if first > 0 and sum(costs[first - 1 : last + 1]) <= budget:
        return first - 1
    return first


def _reference_take_following(
    costs: list[int], first: int, last: int, budget: int
) -> int:
    """Return the following index if adding that slice fits the budget."""
    if last + 1 < len(costs) and sum(costs[first : last + 2]) <= budget:
        return last + 1
    return last


def _reference_window_indices(
    costs: list[int], target: int, budget: int
) -> tuple[int, int]:
    """Model one preceding-then-following round from the public contract."""
    first = last = target
    previous = None
    while previous != (first, last):
        previous = first, last
        first = _reference_take_preceding(costs, first, last, budget)
        last = _reference_take_following(costs, first, last, budget)
    return first, last


@pytest.mark.property
@given(documents_with_target(), BUDGETS)
def test_the_window_always_contains_the_target_sentence(document, budget):
    """
    Whatever the budget, the reference's own sentence is in the context.

    Losing it would hand the encoder text that does not contain the place
    name it is being asked to disambiguate.
    """
    sentences, target = document

    window = expand_window(sentences, target, budget)

    assert sentences[target] in window


@pytest.mark.property
@given(documents_with_target(), BUDGETS)
def test_the_window_is_a_contiguous_run_in_document_order(document, budget):
    """The context is a slice of the document, not a selection from it."""
    sentences, target = document

    window = expand_window(sentences, target, budget)

    first = sentences.index(window[0])
    assert window == sentences[first : first + len(window)]


@pytest.mark.property
@given(documents_with_target(), BUDGETS)
def test_the_window_fits_the_budget_unless_the_target_alone_cannot(document, budget):
    """
    The only context allowed to exceed the budget is a single oversized
    sentence, which the encoder will truncate rather than lose entirely.
    """
    sentences, target = document

    window = expand_window(sentences, target, budget)

    total = sum(sentence.cost for sentence in window)
    assert total <= budget or window == [sentences[target]]


@pytest.mark.property
@given(documents_with_target(), BUDGETS)
def test_neither_neighbour_of_the_window_would_have_fitted(document, budget):
    """
    The window is maximal: it stopped because nothing more fitted.

    Without this, a window could satisfy every other property by simply being
    smaller than it needed to be, wasting the budget the encoder was given.
    """
    sentences, target = document

    window = expand_window(sentences, target, budget)

    spent = sum(sentence.cost for sentence in window)
    remaining = budget - spent
    first = sentences.index(window[0])
    last = first + len(window) - 1
    neighbours = [
        sentences[index]
        for index in (first - 1, last + 1)
        if 0 <= index < len(sentences)
    ]
    assert all(neighbour.cost > remaining for neighbour in neighbours)


@pytest.mark.property
@given(documents_with_target(), BUDGETS)
def test_window_matches_the_documented_greedy_reference(document, budget):
    """The result follows the preceding-then-following contract."""
    sentences, target = document

    first, last = _reference_window_indices(
        [sentence.cost for sentence in sentences], target, budget
    )
    window = expand_window(sentences, target, budget)

    assert window == sentences[first : last + 1]


@pytest.mark.property
@pytest.mark.parametrize(
    (
        "costs",
        "target",
        "smaller_budget",
        "larger_budget",
        "smaller_indices",
        "larger_indices",
    ),
    [
        (
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 2],
            8,
            1,
            2,
            tuple(range(9)),
            (7, 8, 9),
        ),
        ([1, 1, 1, 1, 3], 3, 4, 5, (0, 1, 2, 3), (2, 3, 4)),
    ],
)
def test_greedy_expansion_can_return_fewer_sentences_at_a_larger_budget(
    costs, target, smaller_budget, larger_budget, smaller_indices, larger_indices
):
    """Pin examples where more budget changes the greedy selection order."""
    sentences = _document(costs)
    smaller = expand_window(sentences, target, smaller_budget)
    larger = expand_window(sentences, target, larger_budget)

    assert tuple(sentences.index(sentence) for sentence in smaller) == smaller_indices
    assert tuple(sentences.index(sentence) for sentence in larger) == larger_indices
    assert len(larger) < len(smaller)


@pytest.mark.property
@given(documents_with_target())
def test_every_sentence_is_locatable_from_any_offset_it_covers(document):
    """A reference anywhere inside a sentence resolves to that sentence."""
    sentences, target = document
    sentence = sentences[target]

    for offset in (sentence.start, sentence.end - 1):
        assert locate_sentence(sentences, offset, offset + 1) == target


@pytest.mark.property
@given(documents_with_target(), BUDGETS)
def test_the_context_is_the_window_joined_by_single_spaces(document, budget):
    """select_context is expand_window plus a join, and nothing else."""
    sentences, target = document
    sentence = sentences[target]

    context = select_context(sentences, sentence.start, sentence.end, budget)

    window = expand_window(sentences, target, budget)
    assert context == " ".join(item.text for item in window)


@pytest.mark.property
@given(documents_with_target(), BUDGETS)
def test_the_context_always_contains_the_reference_text(document, budget):
    """The place name the caller asked about is in the string it gets back."""
    sentences, target = document
    sentence = sentences[target]

    context = select_context(sentences, sentence.start, sentence.end, budget)

    assert sentence.text in context


@pytest.mark.property
@given(
    st.lists(st.integers(min_value=0, max_value=20), min_size=1, max_size=8),
    st.integers(min_value=0, max_value=60),
)
def test_an_offset_past_the_document_is_always_rejected(costs, budget):
    """A span beyond the text is reported rather than mapped to a sentence."""
    sentences = _document(costs)
    past_end = sentences[-1].end + 1

    with pytest.raises(ValueError):
        select_context(sentences, past_end, past_end + 1, budget)
