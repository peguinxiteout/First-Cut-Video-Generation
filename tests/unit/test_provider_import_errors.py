from __future__ import annotations

import sys

import pytest

from src.core.exceptions import ModelProviderError


def test_missing_google_genai_package_raises_a_clear_actionable_error(monkeypatch):
    """Regression test for a real production incident: 'from google import
    genai' failing (google-genai not actually installed/importable in the
    deployed environment, even though the app otherwise starts fine, since
    this import is deliberately lazy) used to surface as Python's raw,
    genuinely confusing 'cannot import name genai from google (unknown
    location)' directly in the UI. It must now be a clear, actionable
    ModelProviderError instead."""
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "fake-key-for-this-test", raising=False)
    # The standard trick for making an import fail deterministically in a
    # test: sys.modules[name] = None causes any subsequent `import name`
    # (or `from name import ...`) to raise ImportError immediately.
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.genai", None)

    from src.models.gemini_provider import GeminiProvider

    with pytest.raises(ModelProviderError, match="google-genai"):
        GeminiProvider(model="gemini-3.6-flash")


def test_missing_openai_package_raises_a_clear_actionable_error(monkeypatch):
    """Same fix, applied consistently to the OpenRouter fallback provider,
    which has the identical lazy-import pattern for the openai SDK."""
    monkeypatch.setattr("src.core.config.settings.openrouter_api_key", "fake-key-for-this-test", raising=False)
    monkeypatch.setitem(sys.modules, "openai", None)

    from src.models.openrouter_provider import OpenRouterProvider

    with pytest.raises(ModelProviderError, match="openai"):
        OpenRouterProvider(model="openrouter/free")
