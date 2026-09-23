"""
Tests for reading HIPE-2022 TSV files.

HIPE ships tokens, not text, so the parser rebuilds the text and derives each
toponym's character offsets from it. The offsets are the part worth pinning:
a span that does not select its own surface form is scored as a miss by every
pipeline alike.
"""

from scripts.benchmark.hipe import hipe_qids, parse_hipe

HEADER = (
    "TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tNE-FINE-LIT\tNE-FINE-METO\t"
    "NE-FINE-COMP\tNE-NESTED\tNEL-LIT\tNEL-METO\tMISC\n"
)


def row(token, tag="O", qid="_", misc="_"):
    """Return one TSV row with only the columns the parser reads filled in."""
    return f"{token}\t{tag}\tO\tO\tO\tO\tO\t{qid}\t_\t{misc}\n"


def document(identifier, *rows):
    """Return one HIPE document block."""
    return (
        f"# hipe2022:document_id = {identifier}\n"
        "# hipe2022:language = fr\n" + "".join(rows)
    )


TSV = (
    HEADER
    + document(
        "doc-1",
        row("Il"),
        row("part"),
        row("de"),
        row("Saint", "B-loc", "Q1"),
        row("-", "I-loc", "Q1", "NoSpaceAfter"),
        row("Gall", "I-loc", "Q1", "NoSpaceAfter"),
        row(","),
        row("puis"),
        row("Paris", "B-loc", "Q90", "NoSpaceAfter"),
        row(".", misc="EndOfSentence"),
    )
    + document(
        "doc-2",
        row("Jean", "B-pers", "NIL"),
        row("vit"),
        row("à"),
        row("Genève", "B-loc", "Q71", "NoSpaceAfter"),
        row("."),
    )
)

COORDINATES = {"Q1": (47.42, 9.37), "Q90": (48.85, 2.35), "Q71": (46.2, 6.14)}


def write(tmp_path, text):
    """Write a TSV file and return its path."""
    path = tmp_path / "hipe.tsv"
    path.write_text(text, encoding="utf-8")
    return path


class TestParseHipe:
    """Turning token rows into documents with gold spans."""

    def test_rebuilds_text_honouring_no_space_after(self, tmp_path):
        """Test that tokens are joined as the MISC column says."""
        documents = parse_hipe(write(tmp_path, TSV), COORDINATES)

        assert documents[0].text == "Il part de Saint -Gall, puis Paris."
        assert documents[1].text == "Jean vit à Genève."

    def test_spans_select_their_surface_form(self, tmp_path):
        """Test that every gold span indexes its own name."""
        for doc in parse_hipe(write(tmp_path, TSV), COORDINATES):
            for span in doc.gold:
                assert doc.text[span.start : span.end] == span.name

    def test_keeps_only_locations_with_coordinates(self, tmp_path):
        """Test that persons and unlinked places are not toponyms."""
        documents = parse_hipe(write(tmp_path, TSV), COORDINATES)

        assert [(s.name, s.latitude, s.longitude) for s in documents[0].gold] == [
            ("Saint -Gall", 47.42, 9.37),
            ("Paris", 48.85, 2.35),
        ]
        assert [s.name for s in documents[1].gold] == ["Genève"]

    def test_uses_the_document_id(self, tmp_path):
        """Test that identifiers come from the corpus, not the position."""
        documents = parse_hipe(write(tmp_path, TSV), COORDINATES)

        assert [d.identifier for d in documents] == ["doc-1", "doc-2"]

    def test_drops_a_location_without_coordinates(self, tmp_path):
        """Test that a QID Wikidata gave no coordinates is not scored."""
        coordinates = {k: v for k, v in COORDINATES.items() if k != "Q90"}

        (first, _) = parse_hipe(write(tmp_path, TSV), coordinates)

        assert [s.name for s in first.gold] == ["Saint -Gall"]

    def test_drops_a_location_linked_to_nil(self, tmp_path):
        """Test that an unlinked place cannot be scored by distance."""
        text = HEADER + document("d", row("Ailleurs", "B-loc", "NIL"))

        assert parse_hipe(write(tmp_path, text), COORDINATES) == []

    def test_consecutive_b_tags_are_separate_spans(self, tmp_path):
        """Test that a B tag closes the previous entity."""
        text = HEADER + document(
            "d", row("Paris", "B-loc", "Q90"), row("Genève", "B-loc", "Q71")
        )

        (doc,) = parse_hipe(write(tmp_path, text), COORDINATES)

        assert [s.name for s in doc.gold] == ["Paris", "Genève"]

    def test_limit_stops_after_the_requested_documents(self, tmp_path):
        """Test that a limit shortens the run."""
        assert len(parse_hipe(write(tmp_path, TSV), COORDINATES, limit=1)) == 1

    def test_ignores_comment_lines_inside_a_document(self, tmp_path):
        """Test that segment metadata does not become a token."""
        text = HEADER + document(
            "d", row("Paris", "B-loc", "Q90"), "# segment_iiif_link = x\n", row("!")
        )

        (doc,) = parse_hipe(write(tmp_path, text), COORDINATES)

        assert doc.text == "Paris !"


class TestHipeQids:
    """Listing which Wikidata items need coordinates."""

    def test_lists_linked_locations_only(self, tmp_path):
        """Test that persons and NIL links are not looked up."""
        assert hipe_qids(write(tmp_path, TSV)) == {"Q1", "Q90", "Q71"}


class TestTagCase:
    """NewsEye and TopRes write their tags upper case."""

    def test_reads_upper_case_location_tags(self, tmp_path):
        """Test that B-LOC and I-LOC are locations too."""
        text = HEADER + document(
            "d",
            row("New", "B-LOC", "Q60"),
            row("York", "I-LOC", "Q60"),
            row("Roosevelt", "B-PER", "Q8007"),
        )

        (doc,) = parse_hipe(write(tmp_path, text), {"Q60": (40.7, -74.0)})

        assert [s.name for s in doc.gold] == ["New York"]
