import unicodedata

import pytest

from scripts.masakhaner_benchmark import data
from scripts.masakhaner_benchmark.data import (
    Sentence,
    example_from_sentence,
    examples_from_text,
    git_blob_sha1,
    location_spans,
    parse_sentences,
    split_for,
    token_offsets,
)

# Built from code points so the combining marks are visible in the source.
YORUBA_TOKEN = "Ọ̀yọ́"


def test_label_set_matches_the_upstream_loader_order():
    assert data.LABEL_NAMES == (
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-DATE",
        "I-DATE",
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
        (b"hello\n", "ce013625030ba8dba906f756967f9e9ca394464a"),
    ],
)
def test_git_blob_id_matches_git_for_known_content(content, expected):
    assert git_blob_sha1(content) == expected


def test_sentences_are_separated_by_blank_lines_and_the_last_one_is_kept():
    sentences = parse_sentences(["Ibadan B-LOC", "is O", "", "Kano B-LOC"])

    assert sentences == [
        Sentence(("Ibadan", "is"), ("B-LOC", "O")),
        Sentence(("Kano",), ("B-LOC",)),
    ]


def test_repeated_blank_lines_do_not_create_empty_sentences():
    sentences = parse_sentences(["", "Kano B-LOC", "", "", "", "Ibadan O", ""])

    assert [sentence.tokens for sentence in sentences] == [("Kano",), ("Ibadan",)]


def test_carriage_returns_are_removed_from_the_tag():
    sentences = parse_sentences(["Lagos B-LOC\r", "\r", "Abuja O\r"])

    assert sentences == [
        Sentence(("Lagos",), ("B-LOC",)),
        Sentence(("Abuja",), ("O",)),
    ]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("Lagos", "must hold one token and one tag"),
        ("Lagos B-LOC extra", "must hold one token and one tag"),
        ("Lagos  B-LOC", "must hold one token and one tag"),
        (" B-LOC", "must hold one token and one tag"),
        ("Lagos B-CITY", "unknown tag"),
    ],
)
def test_malformed_token_lines_fail_with_their_line_number(line, message):
    with pytest.raises(ValueError, match=f"Line 2 .*{message}"):
        parse_sentences(["Kano B-LOC", line])


def test_location_tags_join_into_exact_character_spans():
    tokens = ("New", "York", "and", "Rio", "de", "Janeiro")
    tags = ("B-LOC", "I-LOC", "O", "B-LOC", "I-LOC", "I-LOC")

    example = example_from_sentence("por", Sentence(tokens, tags))

    assert example.text == "New York and Rio de Janeiro"
    assert example.gold_spans == frozenset({(0, 8), (13, 27)})
    assert example.malformed_location_tags == 0


def test_offsets_count_code_points_and_the_text_is_not_normalized():
    assert len(YORUBA_TOKEN) == 7
    assert unicodedata.normalize("NFC", YORUBA_TOKEN) != YORUBA_TOKEN

    example = example_from_sentence(
        "yor", Sentence((YORUBA_TOKEN, "ilu", "ni"), ("B-LOC", "O", "O"))
    )

    assert example.text == f"{YORUBA_TOKEN} ilu ni"
    assert example.gold_spans == frozenset({(0, 7)})


def test_a_line_separator_inside_a_token_does_not_split_the_sentence():
    examples = examples_from_text("hau", "a\u2028b B-LOC\nKano O\n")

    assert len(examples) == 1
    assert examples[0].text == "a\u2028b Kano"
    assert examples[0].gold_spans == frozenset({(0, 3)})


def test_examples_are_built_per_blank_line_separated_sentence():
    examples = examples_from_text("hau", "Kano B-LOC\nis O\n\nLagos B-LOC\n")

    assert [example.text for example in examples] == ["Kano is", "Lagos"]
    assert [example.gold_spans for example in examples] == [
        frozenset({(0, 4)}),
        frozenset({(0, 5)}),
    ]


def test_adjacent_begin_tags_start_separate_spans():
    tokens = ("Abuja", "Lagos")
    spans, malformed = location_spans(("B-LOC", "B-LOC"), token_offsets(tokens))

    assert spans == {(0, 5), (6, 11)}
    assert malformed == 0


def test_an_orphan_inside_tag_starts_a_span_and_is_counted():
    tokens = ("a", "b", "c", "d")
    tags = ("O", "I-LOC", "I-LOC", "B-PER")

    spans, malformed = location_spans(tags, token_offsets(tokens))

    assert spans == {(2, 5)}
    assert malformed == 1


def test_date_and_person_tags_are_not_locations():
    example = example_from_sentence(
        "hau", Sentence(("Monday", "Lagos", "Ada"), ("B-DATE", "B-LOC", "B-PER"))
    )

    assert example.gold_spans == frozenset({(7, 12)})


def test_development_and_evaluation_use_different_splits():
    assert split_for("development") == "validation"
    assert split_for("evaluation") == "test"


def test_the_training_split_is_never_available_for_scoring():
    with pytest.raises(ValueError, match="training split is never used"):
        split_for("train")
