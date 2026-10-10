"""Independent UNER fixtures preserve the source sentence and split boundaries."""

import pytest

from scripts.uner_benchmark.data import parse_iob2


def sentence(text, rows, identifier="s1", document="d1"):
    return (
        f"# newdoc id = {document}\n# sent_id = {identifier}\n# text = {text}\n"
        + "\n".join(rows)
    )


def parse(text, *, split="test"):
    return parse_iob2(text, language="pt", configuration="pt_fixture", split=split)


def test_original_unicode_spacing_and_location_labels():
    source = sentence(
        "😀  São\u00a0Paulo, e\u0301 Paris!  ",
        [
            "1\t😀\tO\t-\t-",
            "2\tSão\tB-LOC\t-\tA",
            "3\tPaulo\tI-LOC\t-\tA",
            "4\t,\tO\t-\t-",
            "5\te\u0301\tB-PER\t-\tA",
            "6\tParis\tB-LOC\t-\tA",
            "7\t!\tO\t-\t-",
        ],
    )
    corpus = parse(source)
    record = corpus.sentences[0]
    assert (record.example.text, record.example.gold_spans) == (
        "😀  São\u00a0Paulo, e\u0301 Paris!  ",
        {(3, 12), (17, 22)},
    )
    assert (record.example.language, record.example.malformed_location_tags) == (
        "pt",
        0,
    )
    assert (record.sentence_id, record.document_id, record.identifier) == (
        "s1",
        "d1",
        "pt_fixture/test/s1",
    )
    assert (
        corpus.sentence_count,
        corpus.document_count,
        corpus.token_count,
        corpus.location_count,
        corpus.empty_text_count,
    ) == (1, 1, 7, 2, 0)


def test_repeated_names_negative_sentences_empty_text_and_eof():
    source = (
        sentence("Paris Paris", ["1\tParis\tB-LOC\t-\tA", "2\tParis\tB-ORG\t-\tA"])
        + "\n\n# sent_id = s2\n# text = nobody\n1\tnobody\tO\t-\t-\n\n"
        + sentence("", [], identifier="empty", document="d2")
    )
    corpus = parse(source)
    assert [s.example.gold_spans for s in corpus.sentences] == [{(0, 5)}, set(), set()]
    assert [s.document_id for s in corpus.sentences] == ["d1", "d1", "d2"]
    assert (
        corpus.sentences[2].example.text,
        corpus.sentence_count,
        corpus.document_count,
    ) == ("", 3, 2)
    assert (corpus.token_count, corpus.location_count, corpus.empty_text_count) == (
        3,
        1,
        1,
    )


def test_token_internal_spaces_and_unspaced_chinese():
    corpus = parse(
        sentence(
            "5 000東京。",
            ["1\t5 000\tO\t-\t-", "2\t東京\tB-LOC\t-\tA", "3\t。\tO\t-\t-"],
        )
    )
    assert corpus.sentences[0].example.gold_spans == {(5, 7)}


def test_split_and_configuration_are_part_of_stable_identity():
    source = sentence("Paris", ["1\tParis\tB-LOC\t-\tA"])
    assert parse(source, split="dev").sentences[0].identifier == "pt_fixture/dev/s1"
    assert parse(source).sentences[0].identifier == "pt_fixture/test/s1"


@pytest.mark.parametrize("split", ["validation", "TEST", "all", ""])
def test_unknown_split_rejected(split):
    with pytest.raises(ValueError, match="split"):
        parse("", split=split)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("# sent_id = a\n1\tParis\tB-LOC\t-\t-", "text"),
        ("# text = Paris\n1\tParis\tB-LOC\t-\t-", "sent_id"),
        (sentence("Paris", ["1\tParis\tB-LOC"]), "five"),
        (sentence("Paris", ["2\tParis\tB-LOC\t-\t-"]), "token id"),
        (sentence("Paris", ["1-2\tParis\tB-LOC\t-\t-"]), "token id"),
        (sentence("Paris", ["1.1\tParis\tB-LOC\t-\t-"]), "token id"),
        (sentence("Paris", ["1\tParis\tB-GPE\t-\t-"]), "label"),
        (sentence("Paris", ["1\tParis\tI-LOC\t-\t-"]), "IOB2"),
        (
            sentence("Paris Paris", ["1\tParis\tB-PER\t-\t-", "2\tParis\tI-LOC\t-\t-"]),
            "IOB2",
        ),
        (
            sentence("Paris Paris", ["1\tParis\tB-LOC\t-\t-", "2\tParis\tI-PER\t-\t-"]),
            "IOB2",
        ),
        (sentence("Paris", ["1\tParís\tB-LOC\t-\t-"]), "align"),
        (sentence("xParis", ["1\tParis\tB-LOC\t-\t-"]), "align"),
        (sentence("Paris!", ["1\tParis\tB-LOC\t-\t-"]), "unmatched"),
        (sentence("Paris", ["1\t\tO\t-\t-"]), "empty token"),
        (sentence("Paris", []), "unmatched"),
        (sentence("", ["1\tParis\tB-LOC\t-\t-"]), "align"),
        (
            sentence("Paris", ["1\tParis\tB-LOC\t-\t-"]) + "\n# sent_id = second",
            "metadata",
        ),
        ("# sent_id = a\n# sent_id = b\n# text = ", "duplicate"),
        ("# sent_id = a\n# text = \n# text = ", "duplicate"),
        (sentence("", [], identifier=""), "sent_id"),
        (sentence("", [], document=""), "document"),
    ],
)
def test_malformed_records_fail_without_silent_repairs(source, message):
    with pytest.raises(ValueError, match=message):
        parse(source)


def test_duplicate_sentence_identifiers_fail():
    source = sentence("", []) + "\n\n" + sentence("", [], document="d2")
    with pytest.raises(ValueError, match="duplicate sentence"):
        parse(source)


def test_iob2_cannot_cross_sentence_boundaries():
    source = (
        sentence("New", ["1\tNew\tB-LOC\t-\t-"])
        + "\n\n"
        + sentence("York", ["1\tYork\tI-LOC\t-\t-"], identifier="s2")
    )
    with pytest.raises(ValueError, match="IOB2"):
        parse(source)


def test_comments_crlf_and_missing_document_use_sentence_id():
    source = "# global.columns = ID FORM NER XNER MISC\r\n\r\n# sent_id = a\r\n# text = Paris\r\n# other = ignored\r\n1\tParis\tB-LOC\t-\t-\r\n\r\n"
    record = parse(source).sentences[0]
    assert record.document_id == "a"
    assert record.example.gold_spans == {(0, 5)}


def test_blank_source_retains_zero_counts():
    corpus = parse("\n\n")
    assert corpus.sentences == ()
    assert (
        corpus.token_count,
        corpus.sentence_count,
        corpus.document_count,
        corpus.location_count,
        corpus.empty_text_count,
    ) == (0, 0, 0, 0, 0)


def test_unicode_line_separators_inside_text_are_not_record_boundaries():
    corpus = parse(
        sentence("Paris\u2028Tokyo", ["1\tParis\tB-LOC\t-\tA", "2\tTokyo\tB-LOC\t-\tA"])
    )
    assert corpus.sentences[0].example.text == "Paris\u2028Tokyo"
    assert corpus.sentences[0].example.gold_spans == {(0, 5), (6, 11)}


def test_leading_and_trailing_source_whitespace_are_preserved():
    corpus = parse(sentence("  Paris  ", ["1\tParis\tB-LOC\t-\tA"]))
    assert corpus.sentences[0].example.text == "  Paris  "
    assert corpus.sentences[0].example.gold_spans == {(2, 7)}


def test_inside_person_and_organization_labels_do_not_make_locations():
    corpus = parse(
        sentence(
            "Ada Lovelace Acme Ltd",
            [
                "1\tAda\tB-PER\t-\tA",
                "2\tLovelace\tI-PER\t-\tA",
                "3\tAcme\tB-ORG\t-\tA",
                "4\tLtd\tI-ORG\t-\tA",
            ],
        )
    )
    assert corpus.location_count == 0
    assert corpus.token_count == 4


def test_comments_without_key_value_pairs_are_ignored():
    assert parse("#\n# ignored\n" + sentence("", [])).empty_text_count == 1


def test_official_swedish_fixture_preserves_literal_release_offsets():
    import hashlib
    from pathlib import Path

    payload = Path("tests/fixtures/uner/sv_pud_first_sentence.iob2").read_bytes()
    assert (
        hashlib.sha256(payload).hexdigest()
        == "50150f727ea89cdc86fd80e1a00fdc61124d27a2977b722a23285f5e64159a83"
    )
    corpus = parse_iob2(
        payload.decode("utf-8"), language="sv", configuration="sv_pud", split="test"
    )
    record = corpus.sentences[0]
    assert (record.sentence_id, record.document_id) == ("n01001-0001", "n01001")
    assert (record.example.gold_spans, corpus.token_count) == ({(69, 72)}, 35)
    assert record.example.text[69:72] == "USA"


def test_document_count_keeps_fallback_and_explicit_ids_separate():
    source = "# sent_id = first\n# text = \n\n# newdoc id = first\n# sent_id = second\n# text = \n"
    corpus = parse(source)
    assert [record.document_id for record in corpus.sentences] == ["first", "first"]
    assert corpus.document_count == 2


def test_bare_document_boundary_is_not_silently_merged():
    source = sentence("", []) + "\n\n# newdoc\n# sent_id = second\n# text = \n"
    with pytest.raises(ValueError, match="document boundary"):
        parse(source)


@pytest.mark.parametrize(
    "comment",
    [
        "# newdoc\tid = b",
        "#\tnewdoc id = b",
        "# newdoc\u00a0id = b",
        "# text\t= wrong",
        "# sent_id\t= other",
    ],
)
def test_malformed_reserved_metadata_is_not_ignored(comment):
    source = sentence("", []) + "\n\n" + comment + "\n# sent_id = second\n# text = \n"
    with pytest.raises(ValueError, match="metadata"):
        parse(source)


def test_release_document_id_allows_absent_space_after_equals():
    source = "# newdoc id =set.sr.11\n# sent_id = s1\n# text = Paris\n1\tParis\tB-LOC\t-\t-\n"
    record = parse(source).sentences[0]
    assert (record.source_document_id, record.document_id) == ("set.sr.11", "set.sr.11")
    assert record.example.gold_spans == {(0, 5)}


def test_released_other_labels_remain_non_location_and_strict_iob2():
    source = sentence(
        "Pomurci New brand Paris",
        [
            "1\tPomurci\tB-OTH\t-\t-",
            "2\tNew\tB-OTH\t-\t-",
            "3\tbrand\tI-OTH\t-\t-",
            "4\tParis\tB-LOC\t-\t-",
        ],
    )
    assert parse(source).sentences[0].example.gold_spans == {(18, 23)}
    with pytest.raises(ValueError, match="IOB2"):
        parse(sentence("Pomurci", ["1\tPomurci\tI-OTH\t-\t-"]))
