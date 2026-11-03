"""One way for blog agents to call a model, with a per-run circuit breaker.

Every blog agent has a deterministic fallback, so a model that is not
there should cost nothing. Before this, it cost a lot: `with_retries`
backs off 2 + 4 + 8 seconds on every ModelProviderError, so a run with no
API key (or an exhausted free-tier quota) spent well over a minute
sleeping across seven agents before producing exactly the post the
fallbacks would have produced at once.

So the first call that fails for good — missing key, both providers
down, quota gone after retries — opens a breaker for the rest of THIS
run (a ContextVar, so concurrent runs in one process do not see each
other's state) and every later agent goes straight to its fallback.

Agents pass their own module's `call_with_fallback` in as `caller`, so
tests that patch `src.agents.blog.<agent>.call_with_fallback` keep
working exactly as before.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Any, Callable

from src.agents.base import with_retries
from src.core.exceptions import ConfigError, ModelProviderError

logger = logging.getLogger(__name__)

# A mutable holder per run. LangGraph runs each node in a COPY of the
# calling context, so a value set inside one node would vanish before the
# next; mutating a dict created before the graph starts does not.
_state: contextvars.ContextVar[dict | None] = contextvars.ContextVar("blog_llm_state", default=None)

# Failures no amount of waiting fixes. Retrying these only adds sleep.
_PERMANENT_MARKERS = ("is not set", "api_key", "api key", "not configured", "could not resolve a model",
                      "unknown provider", "no provider configured", "401", "403", "permission")


class ModelUnavailable(ModelProviderError):
    """Raised without calling anything once the breaker is open."""


def reset() -> contextvars.Token:
    """Start a run with the breaker closed. Returns the token for `restore`."""
    return _state.set({"reason": ""})


def restore(token: contextvars.Token) -> None:
    _state.reset(token)


def unavailable_reason() -> str:
    state = _state.get()
    return state.get("reason", "") if state else ""


def _open(reason: str) -> None:
    state = _state.get()
    if state is not None:
        state["reason"] = reason[:200]


def _is_permanent(exc: Exception) -> bool:
    if isinstance(exc, ConfigError):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _PERMANENT_MARKERS)


def generate_json(role: Any, prompt: str, caller: Callable[..., Any]) -> Any:
    """`caller(role, "generate_json", prompt)` with retries for transient
    errors, no retries for permanent ones, and nothing at all once the
    breaker for this run is open."""
    reason = unavailable_reason()
    if reason:
        raise ModelUnavailable(f"model unavailable for this run: {reason}")
    try:
        return caller(role, "generate_json", prompt)
    except (ModelProviderError, ConfigError) as first:
        if _is_permanent(first):
            _open(str(first))
            raise
    try:
        return with_retries(lambda: caller(role, "generate_json", prompt), max_retries=3)
    except (ModelProviderError, ConfigError) as exc:
        _open(str(exc))
        raise
