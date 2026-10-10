"""Shared fixtures: every test here runs with external network access disabled."""

from collections.abc import Iterator

import pytest

from tests.fixtures.network import external_network_disabled


@pytest.fixture(autouse=True)
def no_external_network() -> Iterator[None]:
    """Fail any test that tries to reach a host, rather than letting it succeed."""
    with external_network_disabled():
        yield
