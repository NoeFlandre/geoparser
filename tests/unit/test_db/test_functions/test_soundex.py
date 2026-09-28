"""
Unit tests for geoparser/db/functions/soundex.py
"""

import pytest

from geoparser.db.functions.soundex import soundex


@pytest.mark.unit
class TestSoundex:
    """Test the soundex() function."""

    def test_encodes_basic_name(self):
        """Test that a basic name is encoded to its Soundex code."""
        assert soundex("Robert") == "R163"

    def test_collapses_adjacent_duplicate_codes(self):
        """Test that adjacent letters with the same code are coded once."""
        assert soundex("Pfister") == "P236"

    def test_treats_h_and_w_as_transparent(self):
        """Test that h and w do not break adjacency of identical codes."""
        assert soundex("Ashcraft") == "A261"

    def test_keeps_codes_separated_by_vowel(self):
        """Test that identical codes separated by a vowel are coded twice."""
        assert soundex("Tymczak") == "T522"

    def test_pads_short_codes_with_zeros(self):
        """Test that short codes are zero-padded to four characters."""
        assert soundex("Lee") == "L000"

    def test_is_case_insensitive(self):
        """Test that encoding is case-insensitive."""
        assert soundex("andorra") == soundex("ANDORRA")

    def test_misspelling_matches_original(self):
        """Test that a misspelling shares the Soundex code of the original."""
        assert soundex("Andora") == soundex("Andorra") == "A536"

    def test_ignores_non_alphabetic_characters(self):
        """Test that non-alphabetic characters are ignored."""
        assert soundex("O'Brien") == soundex("OBrien")

    def test_returns_empty_string_for_empty_input(self):
        """Test that an empty string returns an empty code."""
        assert soundex("") == ""

    def test_returns_empty_string_for_none(self):
        """Test that None returns an empty code."""
        assert soundex(None) == ""

    def test_returns_empty_string_for_non_alphabetic_input(self):
        """Test that a string without letters returns an empty code."""
        assert soundex("123 !!") == ""


@pytest.mark.unit
class TestSoundexReferenceCodes:
    """Published American Soundex values."""

    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("Ashcraft", "A261"),
            ("Tymczak", "T522"),
            ("Pfister", "P236"),
            ("Robert", "R163"),
            ("Rupert", "R163"),
            ("Honeyman", "H555"),
            ("Washington", "W252"),
        ],
    )
    def test_matches_the_published_code(self, name, code):
        """Each reference name encodes to its documented value."""
        assert soundex(name) == code

    def test_treats_w_as_transparent_between_same_coded_letters(self):
        """A w between two same-coded consonants does not break adjacency."""
        assert soundex("Bwb") == "B000"

    def test_a_vowel_between_same_coded_letters_does_break_adjacency(self):
        """A vowel resets adjacency, so the code repeats."""
        assert soundex("Bab") == "B100"

    def test_is_case_insensitive(self):
        assert soundex("ashcraft") == soundex("ASHCRAFT") == "A261"

    def test_ignores_non_alphabetic_characters(self):
        assert soundex("O'Brien-2") == soundex("OBrien")

    @pytest.mark.parametrize("text", ["", "1234", "  "])
    def test_returns_empty_without_letters(self, text):
        assert soundex(text) == ""
