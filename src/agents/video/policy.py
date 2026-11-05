"""What a video run may reach (configs/video/policy.yaml).

A run is limited to the vendors and the language-model roles the policy lists.
Inside `video_run_policy()`:

    - the keys of every other vendor are blanked in Settings, so nothing can
      build a provider for it (the environment's FAL_KEY goes too),
    - the research tools and the image service are switched off, as neither
      is a listed vendor's,
    - the gateway refuses a role that is not listed, and a fallback to a
      vendor that is not (OpenRouter behind a Gemini role), and says so.

What was refused is on the policy that is yielded; the workflow writes it
into the record's notes. An agent whose role is not allowed should not even
ask: `role_allowed(role)` lets it take its own fallback without the retries.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

from src.core.config import load_yaml_config, settings
from src.models import gateway

logger = logging.getLogger(__name__)

# The Settings fields that hold each vendor's key.
VENDOR_KEYS = {
    "gemini": ("gemini_api_key",),
    "google_cloud_tts": ("google_cloud_tts_api_key",),
    "beatoven": ("beatoven_api_key",),
    "openrouter": ("openrouter_api_key",),
    "cloudflare": ("cloudflare_account_id", "cloudflare_api_token"),
    "huggingface": ("hf_token",),
    "tavily": ("tavily_api_key",),
}
_ACTIVE: list[Optional[gateway.CallPolicy]] = []


def _config() -> dict:
    return load_yaml_config("video/policy.yaml") or {}


def active() -> Optional[gateway.CallPolicy]:
    return _ACTIVE[-1] if _ACTIVE else None


def role_allowed(role: str) -> bool:
    """False when a run under the policy may not ask this model role."""
    policy = active()
    return not (policy and policy.roles and role not in policy.roles)


@contextmanager
def video_run_policy() -> Iterator[Optional[gateway.CallPolicy]]:
    cfg = _config()
    vendors, roles = list(cfg.get("allowed_vendors") or []), list(cfg.get("model_roles") or [])
    if not vendors and not roles:
        yield None
        return
    saved = {}
    if vendors:
        for vendor, fields in VENDOR_KEYS.items():
            if vendor not in vendors:
                for field in fields:
                    saved[field] = getattr(settings, field)
                    setattr(settings, field, "")
        for field in ("tools_enabled", "image_gen_provider", "video_gen_provider"):
            saved[field] = getattr(settings, field)
        settings.tools_enabled = False
        settings.image_gen_provider = "mock"
        settings.video_gen_provider = "kenburns"       # no vendor's video model is allowed in this phase
    fal_key = os.environ.pop("FAL_KEY", None) if vendors and "fal" not in vendors else None
    with gateway.restricted(vendors, roles) as policy:
        _ACTIVE.append(policy)
        try:
            yield policy
        finally:
            _ACTIVE.pop()
            for field, value in saved.items():
                setattr(settings, field, value)
            if fal_key is not None:
                os.environ["FAL_KEY"] = fal_key
