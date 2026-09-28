"""Helpers for preventing external network access in offline tests."""

import socket
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from ipaddress import ip_address
from unittest.mock import patch


def _is_loopback_host(host: str | bytes) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    host = host.casefold().rstrip(".")
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _require_loopback_address(address: object) -> None:
    if isinstance(address, (str, bytes)):
        return
    if isinstance(address, tuple) and address and _is_loopback_host(address[0]):
        return
    raise RuntimeError("network access is disabled in offline tests")


@contextmanager
def external_network_disabled() -> Iterator[None]:
    """Block external socket operations until the context exits."""
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and not _is_loopback_host(host):
            raise RuntimeError("network access is disabled in offline tests")
        return original_getaddrinfo(host, *args, **kwargs)

    def connect(sock, address):
        _require_loopback_address(address)
        return original_connect(sock, address)

    def connect_ex(sock, address):
        _require_loopback_address(address)
        return original_connect_ex(sock, address)

    def sendto(sock, data, *args):
        _require_loopback_address(args[-1])
        return original_sendto(sock, data, *args)

    with ExitStack() as stack:
        stack.enter_context(patch.object(socket, "getaddrinfo", getaddrinfo))
        stack.enter_context(patch.object(socket.socket, "connect", connect))
        stack.enter_context(patch.object(socket.socket, "connect_ex", connect_ex))
        stack.enter_context(patch.object(socket.socket, "sendto", sendto))
        yield
