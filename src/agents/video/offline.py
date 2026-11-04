"""When a model is not even tried.

`with_retries` backs off 2 + 4 + 8 seconds on every model error, so asking a
model that has no key just to learn it cannot answer costs well over ten
seconds per agent. An offline run, or a machine with no model key, goes
straight to the agent's own fallback instead."""
from __future__ import annotations

from src.core.config import settings


def model_blocker(offline: bool) -> str:
    """Why no model call should be made, or "" when one may be."""
    if offline:
        return "offline run"
    if not (settings.gemini_api_key or settings.openrouter_api_key):
        return "no model key is set"
    return ""
