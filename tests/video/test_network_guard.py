"""The guard in tests/conftest.py: nothing in the suite reaches a remote host,
and the keys in .env are not in play."""
from __future__ import annotations

import socket

import pytest

from src.core.config import settings


def test_a_connect_to_a_remote_address_is_refused_and_recorded(network_attempts):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="blocked in tests"):
            sock.connect(("203.0.113.5", 443))      # TEST-NET-3: nothing is ever sent
    finally:
        sock.close()
    assert network_attempts == ["203.0.113.5"]
    network_attempts.clear()


def test_connect_ex_is_refused_too(network_attempts):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="blocked in tests"):
            sock.connect_ex(("203.0.113.5", 443))
    finally:
        sock.close()
    network_attempts.clear()


def test_a_hostname_lookup_is_refused(network_attempts):
    with pytest.raises(OSError, match="blocked in tests"):
        socket.create_connection(("api.example.com", 443), timeout=1)
    assert network_attempts == ["api.example.com"]
    network_attempts.clear()


def test_loopback_still_works(network_attempts):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(("127.0.0.1", server.getsockname()[1]), timeout=2)
    client.close()
    server.close()
    assert network_attempts == []


def test_the_model_keys_are_blank_whatever_is_in_the_environment():
    assert settings.gemini_api_key == ""
    assert settings.openrouter_api_key == ""
