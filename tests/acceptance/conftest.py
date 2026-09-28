from collections.abc import Iterator

import pytest

from tests.fixtures.network import external_network_disabled


@pytest.fixture(autouse=True)
def disable_external_network() -> Iterator[None]:
    with external_network_disabled():
        yield
