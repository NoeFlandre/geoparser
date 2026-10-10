"""Independent span, Unicode, noisy-text and invalid-record fixtures for MultiCoNER II."""

import pytest

from scripts.multiconer_benchmark.conll import parse_conll

# Written as code points so the source stays free of ambiguous characters.
LINE_SEPARATOR = chr(0x2028)
NO_BREAK_SPACE = chr(0xA0)

# Hand-derived fixture. Offsets are Python code points in the space-joined text.
MULTI_LANGUAGE = "\n".join(
    [
        "# id zh-1\tdomain=zh",
        "我 _ _ O",
        "在 _ _ O",
        "北京 _ _ B-HumanSettlement",
        "的 _ _ O",
        "故宫 _ _ B-Facility",
        "参观 _ _ O",
        "",
        "# id hi-1\tdomain=hi",
        "नमस्ते _ _ O",
        "दिल्ली _ _ B-HumanSettlement",
        "",
        "# id en-1\tdomain=en",
        "i _ _ O",
        "flew _ _ O",
        "to _ _ O",
        "new _ _ B-HumanSettlement",
        "yrok _ _ I-HumanSettlement",
        "😀 _ _ O",
        "lol _ _ O",
        "",
        "# id fa-1\tdomain=fa",
        "تهران _ _ B-HumanSettlement",
        "ایران _ _ B-HumanSettlement",
        "",
        "# id en-2\tdomain=en",
        "Barcelona _ _ B-SportsGRP",
        "won _ _ O",
        "again _ _ O",
        "",
        "# id fr-1\tdomain=fr",
        "Gare _ _ B-Station",
        "du _ _ I-Station",
        "Nord _ _ I-Station",
        "closed _ _ O",
        "",
    ]
)


def _by_id(parsed):
    return {sentence.sample_id: sentence for sentence in parsed.sentences}


def test_chinese_sentence_has_hand_counted_text_and_location_offsets():
    sentence = _by_id(parse_conll(MULTI_LANGUAGE))["zh-1"]
    assert sentence.text == "我 在 北京 的 故宫 参观"
    assert len(sentence.text) == 14
    assert sentence.offsets[2] == (4, 6)
    assert sentence.location_spans() == {(4, 6), (9, 11)}


def test_devanagari_combining_marks_count_as_code_points():
    sentence = _by_id(parse_conll(MULTI_LANGUAGE))["hi-1"]
    assert sentence.text == "नमस्ते दिल्ली"
    assert len(sentence.text) == 13
    assert sentence.location_spans() == {(7, 13)}


def test_astral_emoji_is_one_character_and_noisy_lowercase_spans_survive():
    sentence = _by_id(parse_conll(MULTI_LANGUAGE))["en-1"]
    assert sentence.text == "i flew to new yrok 😀 lol"
    assert len(sentence.text) == 24
    assert sentence.offsets[5] == (19, 20)
    assert sentence.location_spans() == {(10, 18)}


def test_farsi_right_to_left_tokens_keep_their_offsets():
    sentence = _by_id(parse_conll(MULTI_LANGUAGE))["fa-1"]
    assert sentence.text == "تهران ایران"
    assert sentence.location_spans() == {(0, 5), (6, 11)}


def test_non_place_gold_is_retained_but_excluded_from_location_spans():
    sentence = _by_id(parse_conll(MULTI_LANGUAGE))["en-2"]
    assert [entity.label for entity in sentence.entities] == ["SportsGRP"]
    assert sentence.entities[0].start == 0
    assert sentence.entities[0].end == 9
    assert sentence.location_spans() == frozenset()


def test_station_continuation_tags_form_one_span():
    sentence = _by_id(parse_conll(MULTI_LANGUAGE))["fr-1"]
    assert sentence.text == "Gare du Nord closed"
    assert len(sentence.text) == 19
    assert sentence.location_spans() == {(0, 12)}


def test_clean_fixture_has_no_invalid_records_and_counts_every_sentence():
    parsed = parse_conll(MULTI_LANGUAGE)
    assert parsed.invalid == ()
    assert len(parsed.sentences) == 6
    assert parsed.record_count == 6


def test_crlf_and_bom_are_removed_before_parsing():
    text = "\ufeff# id c\tdomain=en\r\nParis _ _ B-HumanSettlement\r\n\r\n"
    parsed = parse_conll(text)
    assert parsed.invalid == ()
    assert parsed.sentences[0].tokens == ("Paris",)
    assert parsed.sentences[0].location_spans() == {(0, 5)}


def test_line_separator_inside_a_token_is_kept_literally():
    separator = parse_conll(f"# id ls\tdomain=en\na{LINE_SEPARATOR}b _ _ O\n")
    assert separator.invalid == ()
    assert separator.sentences[0].tokens == (f"a{LINE_SEPARATOR}b",)
    assert separator.sentences[0].text == f"a{LINE_SEPARATOR}b"


def test_final_sentence_without_blank_line_is_kept():
    parsed = parse_conll("# id last\tdomain=en\nOttawa _ _ B-HumanSettlement")
    assert [sentence.sample_id for sentence in parsed.sentences] == ["last"]


def test_a_token_with_spaces_stays_one_token():
    parsed = parse_conll(
        "# id s1\tdomain=en\nNew York _ _ B-HumanSettlement\nRome _ _ O\n"
    )
    assert parsed.invalid == ()
    assert parsed.sentences[0].tokens == ("New York", "Rome")
    assert parsed.sentences[0].text == "New York Rome"


def test_a_token_with_spaces_has_exact_offsets_and_location_span():
    sentence = parse_conll(
        "# id s1\tdomain=en\nNew York _ _ B-HumanSettlement\nRome _ _ O\n"
    ).sentences[0]
    assert sentence.offsets == ((0, 8), (9, 13))
    assert sentence.location_spans() == {(0, 8)}


def test_spaced_tokens_continue_an_entity_across_their_spaces():
    parsed = parse_conll(
        "# id s2\tdomain=en\n"
        "New York _ _ B-HumanSettlement\n"
        "City _ _ I-HumanSettlement\n"
    )
    sentence = parsed.sentences[0]
    assert sentence.text == "New York City"
    assert sentence.location_spans() == {(0, 13)}


def test_a_token_that_ends_in_an_underscore_keeps_it():
    parsed = parse_conll("# id s3\tdomain=en\nMr _ _ _ O\n")
    assert parsed.sentences[0].tokens == ("Mr _",)


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        (
            "New York _ _ B-HumanSettlement\textra",
            "wrong column count: expected 4, found 6",
        ),
        ("Lima x x B-HumanSettlement", "invalid separator columns"),
    ],
)
def test_token_lines_with_bad_columns_are_invalid(line, reason):
    parsed = parse_conll(f"# id bad\tdomain=en\n{line}\n")
    assert [item.reason for item in parsed.invalid] == [reason]


def test_a_sample_id_with_spaces_is_kept_whole():
    parsed = parse_conll("# id sample one 2\tdomain=en\nRome _ _ O\n")
    assert [sentence.sample_id for sentence in parsed.sentences] == ["sample one 2"]


def test_sample_id_keeps_its_leading_spaces_and_trailing_spaces_after_the_domain_are_ignored():
    parsed = parse_conll("# id  x \tdomain=en  \nRome _ _ O\n")
    assert [sentence.sample_id for sentence in parsed.sentences] == [" x"]


@pytest.mark.parametrize(
    ("header", "sample_id", "reason"),
    [
        ("# id \tdomain=en", None, "missing sample id"),
        ("# id x\tdomain=", "x", "missing domain"),
        ("# id x extra", "x", "missing domain"),
    ],
)
def test_header_defects_are_reported_with_their_sample_id(header, sample_id, reason):
    parsed = parse_conll(f"{header}\nRome _ _ O\n")
    assert [(item.sample_id, item.reason) for item in parsed.invalid] == [
        (sample_id, reason)
    ]


INVALID_LINES = [
    "# id ok-1\tdomain=en",  # 1
    "Paris _ _ B-HumanSettlement",  # 2
    "",  # 3
    "Rome _ _ B-HumanSettlement",  # 4 missing header
    "",  # 5
    "# id bad-col\tdomain=en",  # 6
    "Oslo _ _ B-HumanSettlement extra",  # 7 five columns
    "",  # 8
    "# id bad-label\tdomain=en",  # 9
    "Mars _ _ B-Planet",  # 10 unknown type
    "",  # 11
    "# id orphan\tdomain=en",  # 12
    "Lyon _ _ I-HumanSettlement",  # 13 orphan I-
    "",  # 14
    "# id switch\tdomain=en",  # 15
    "Gare _ _ B-Station",  # 16
    "du _ _ I-Facility",  # 17 type switch
    "",  # 18
    "# id ok-1\tdomain=en",  # 19 duplicate id
    "Nice _ _ B-HumanSettlement",  # 20
    "",  # 21
    "# id empty\tdomain=en",  # 22
    "",  # 23 empty sentence
    "# id nodomain",  # 24
    "Bern _ _ B-HumanSettlement",  # 25
    "",  # 26
    "# id sep\tdomain=en",  # 27
    "Lima x _ B-HumanSettlement",  # 28 separator column
    "",  # 29
    "# id nbsp\tdomain=en",  # 30
    f"Quito{NO_BREAK_SPACE}_ _ B-HumanSettlement",  # 31 NBSP is not a column separator
    "",  # 32
    "# id last\tdomain=en",  # 33
    "Ottawa _ _ B-HumanSettlement",  # 34
]


def test_valid_sentences_survive_around_rejected_blocks():
    parsed = parse_conll("\n".join(INVALID_LINES))
    assert [sentence.sample_id for sentence in parsed.sentences] == ["ok-1", "last"]


def test_every_invalid_record_is_counted_with_its_line_and_reason():
    parsed = parse_conll("\n".join(INVALID_LINES))
    observed = [(item.line, item.sample_id, item.reason) for item in parsed.invalid]
    assert observed == [
        (4, None, "missing sentence header"),
        (7, "bad-col", "wrong column count: expected 4, found 5"),
        (10, "bad-label", "unknown entity type: Planet"),
        (13, "orphan", "I- tag does not continue an entity"),
        (17, "switch", "I- tag does not continue an entity"),
        (19, "ok-1", "duplicate sample id"),
        (22, "empty", "empty sentence"),
        (24, "nodomain", "missing domain"),
        (28, "sep", "invalid separator columns"),
        (31, "nbsp", "wrong column count: expected 4, found 3"),
    ]
    assert parsed.record_count == 12
    assert len(parsed.sentences) + len(parsed.invalid) == parsed.record_count


def test_expected_domain_rejects_a_sentence_from_another_language():
    text = "# id x\tdomain=fr\nTokyo _ _ B-HumanSettlement\n"
    parsed = parse_conll(text, expected_domain="en")
    assert parsed.sentences == ()
    assert [(item.line, item.reason) for item in parsed.invalid] == [
        (1, "domain mismatch: expected en, found fr")
    ]
