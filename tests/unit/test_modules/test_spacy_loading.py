"""The shared load-or-download helper for spaCy pipelines."""

from unittest.mock import Mock, call, patch

import pytest

from geoparser.modules._spacy import load_spacy_model


@pytest.mark.unit
class TestLoadSpacyModel:
    @patch("geoparser.modules._spacy.spacy.cli.download")
    @patch("geoparser.modules._spacy.spacy.load")
    def test_loads_installed_model(self, mock_load, mock_download):
        nlp = Mock()
        mock_load.return_value = nlp

        assert load_spacy_model("en_core_web_sm") is nlp
        mock_download.assert_not_called()

    @patch("geoparser.modules._spacy.spacy.cli.download")
    @patch("geoparser.modules._spacy.spacy.load")
    def test_downloads_missing_model_and_reloads(
        self, mock_load, mock_download, caplog
    ):
        nlp = Mock()
        mock_load.side_effect = [OSError("missing"), nlp]

        with caplog.at_level("INFO", logger="geoparser"):
            assert load_spacy_model("en_core_web_sm") is nlp

        mock_download.assert_called_once_with("en_core_web_sm")
        assert mock_load.call_args_list == [
            call("en_core_web_sm"),
            call("en_core_web_sm"),
        ]
        assert "Downloading spaCy model 'en_core_web_sm'..." in caplog.messages
