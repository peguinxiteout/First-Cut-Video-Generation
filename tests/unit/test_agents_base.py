from __future__ import annotations

import pytest

from src.agents.base import with_retries
from src.core.exceptions import ModelProviderError


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    # Every test here exercises retry/backoff logic, not real wall-clock time.
    monkeypatch.setattr("src.agents.base.time.sleep", lambda *_args, **_kwargs: None)


def test_with_retries_returns_immediately_on_success():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert with_retries(fn) == "ok"
    assert len(calls) == 1


def test_with_retries_retries_then_succeeds():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ModelProviderError("still failing")
        return "ok"

    assert with_retries(fn) == "ok"
    assert attempts["n"] == 3


def test_with_retries_raises_after_max_retries():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        raise ModelProviderError("still failing")

    with pytest.raises(ModelProviderError):
        with_retries(fn, max_retries=2)
    assert attempts["n"] == 2


def test_with_retries_gives_up_immediately_on_non_provider_errors():
    """Anything that isn't a ModelProviderError (e.g. a bug in our own
    code) should propagate immediately, not get masked by retry/backoff."""

    def fn():
        raise ValueError("this is a real bug, not a transient API error")

    with pytest.raises(ValueError):
        with_retries(fn)
