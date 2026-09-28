import socket

import pytest

from tests.fixtures.network import external_network_disabled


def test_external_network_guard_restores_socket_functions() -> None:
    assert socket.getaddrinfo.__module__ == "socket"
    original_getaddrinfo = socket.getaddrinfo

    with external_network_disabled():
        assert socket.getaddrinfo is not original_getaddrinfo
        with pytest.raises(RuntimeError, match="network access is disabled"):
            socket.getaddrinfo("203.0.113.1", 443)

    assert socket.getaddrinfo is original_getaddrinfo
