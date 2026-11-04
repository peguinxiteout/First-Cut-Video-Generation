from __future__ import annotations

import pytest

from tests.video.fakes import make_photo


@pytest.fixture(autouse=True)
def _no_vendor_keys(monkeypatch):
    """The providers read their keys from the process environment as well as
    from Settings; a developer's own FAL_KEY must not make a paid provider
    look available in a test."""
    monkeypatch.delenv("FAL_KEY", raising=False)


@pytest.fixture
def photo(tmp_path):
    return make_photo(tmp_path / "photos" / "wide.jpg", (1600, 900), seed=1)


@pytest.fixture
def portrait_photo(tmp_path):
    return make_photo(tmp_path / "photos" / "tall.jpg", (900, 1600), seed=2)


@pytest.fixture(autouse=True)
def _tests_use_the_mock_voice(monkeypatch):
    """The configured chain ends in a voice on this machine (Windows SAPI). A
    test that renders a video must not depend on one, so the default speech
    provider here is the mock: audible, deterministic, instant. Tests of the
    real chain build their own service."""
    monkeypatch.setattr("src.core.config.settings.speech_provider", "mock", raising=False)
