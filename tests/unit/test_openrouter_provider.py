from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.exceptions import ModelProviderError
from src.models.openrouter_provider import _extract_message_content


def test_extract_message_content_happy_path():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello", refusal=None))]
    )
    assert _extract_message_content(response, "ctx") == "hello"


def test_extract_message_content_raises_clean_error_on_none_choices():
    """Regression test: this exact shape (choices=None) previously escaped
    as a raw, unretriable 'NoneType' object is not subscriptable' from
    response.choices[0] with no try/except around it. Some free-tier/newer
    models occasionally return responses shaped like this."""
    response = SimpleNamespace(choices=None)
    with pytest.raises(ModelProviderError, match="no choices"):
        _extract_message_content(response, "ctx")


def test_extract_message_content_raises_clean_error_on_empty_choices():
    response = SimpleNamespace(choices=[])
    with pytest.raises(ModelProviderError, match="no choices"):
        _extract_message_content(response, "ctx")


def test_extract_message_content_raises_clean_error_on_none_message():
    response = SimpleNamespace(choices=[SimpleNamespace(message=None)])
    with pytest.raises(ModelProviderError, match="no message"):
        _extract_message_content(response, "ctx")


def test_extract_message_content_raises_clean_error_on_refusal():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="policy violation"))]
    )
    with pytest.raises(ModelProviderError, match="refused"):
        _extract_message_content(response, "ctx")


def test_extract_message_content_raises_clean_error_on_empty_content():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", refusal=None))]
    )
    with pytest.raises(ModelProviderError, match="empty response content"):
        _extract_message_content(response, "ctx")


def test_all_malformed_response_errors_are_retriable_model_provider_errors():
    """The whole point of this fix: every failure path here must raise
    ModelProviderError specifically, since that's the only exception type
    with_retries() (src/agents/base.py) catches and retries."""
    from src.agents.base import with_retries

    attempts = {"n": 0}

    def flaky_call():
        attempts["n"] += 1
        if attempts["n"] < 2:
            return _extract_message_content(SimpleNamespace(choices=None), "ctx")
        return "recovered"

    # Disable real sleeping for this test.
    import src.agents.base as base_module

    original_sleep = base_module.time.sleep
    base_module.time.sleep = lambda *_a, **_k: None
    try:
        assert with_retries(flaky_call) == "recovered"
    finally:
        base_module.time.sleep = original_sleep
