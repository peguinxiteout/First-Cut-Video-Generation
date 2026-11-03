from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from src.core.exceptions import ModelProviderError

T = TypeVar("T")

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2.0
RATE_LIMIT_BACKOFF_SECONDS = 5.0  # rate-limit/quota errors get a slower, more patient backoff
MAX_BACKOFF_SECONDS = 25.0


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("429", "rate limit", "rate-limited", "quota", "rate_limit"))


def with_retries(fn: Callable[[], T], *, max_retries: int = MAX_RETRIES) -> T:
    """Small shared retry helper for agent calls that hit a model provider.
    Retries on ModelProviderError with exponential backoff + jitter;
    re-raises the last error if all attempts fail so the workflow node can
    decide what to do (see workflows/brochure.py error handling).

    Rate-limit/quota errors (HTTP 429, "temporarily rate-limited", etc.) get
    a slower backoff than other transient errors — free-tier congestion on
    a shared pool ("retry shortly") often takes longer than a network blip
    to clear, and retrying too fast just adds to the congestion."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except ModelProviderError as exc:
            last_exc = exc
            if attempt < max_retries:
                base = RATE_LIMIT_BACKOFF_SECONDS if _is_rate_limit_error(exc) else BASE_BACKOFF_SECONDS
                delay = min(base * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS) + random.uniform(0, 1.0)
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc
