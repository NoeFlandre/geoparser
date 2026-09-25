"""Shared spaCy model loading for the modules that need a pipeline."""

import spacy
import spacy.cli

from geoparser._logging import get_logger

logger = get_logger(__name__)


def load_spacy_model(model_name: str) -> spacy.language.Language:
    """
    Load a spaCy model, downloading it if necessary.

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        Loaded spaCy Language model
    """
    try:
        return spacy.load(model_name)
    except OSError:
        # Model not found, download it
        # pragma: no mutate start - progress prose, not behaviour; the
        # download and reload below are what the tests pin.
        logger.info(f"Downloading spaCy model '{model_name}'...")
        # pragma: no mutate end
        spacy.cli.download(model_name)
        return spacy.load(model_name)
