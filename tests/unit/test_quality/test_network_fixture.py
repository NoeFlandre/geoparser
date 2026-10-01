"""Tests for the offline test socket-address guard."""

import pytest

from tests.fixtures.network import _require_loopback_address


@pytest.mark.parametrize(
    "address",
    [
        "/tmp/test.sock",
        b"\x00test",
        ("localhost", 1234),
        ("127.0.0.1", 1234),
        (b"::1", 1234),
    ],
)
def test_allows_local_socket_addresses(address: object) -> None:
    _require_loopback_address(address)


@pytest.mark.parametrize(
    "address",
    [
        ("example.com", 443),
        ("203.0.113.1", 443),
        (),
        None,
    ],
)
def test_rejects_non_loopback_socket_addresses(address: object) -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        _require_loopback_address(address)
