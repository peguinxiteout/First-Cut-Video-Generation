"""
The model gateway is the ONLY place that knows how a logical role
(e.g. "copywriter") maps to a concrete provider+model. Agents ask the
gateway for a provider by role; if the primary provider errors, the
gateway transparently retries once against the configured fallback.

    provider = get_provider_for_role("copywriter")
    copy = provider.generate_json(prompt)

Change configs/models.yaml to point a role at a different provider/model —
no agent code changes.
"""
from __future__ import annotations

import os
from functools import lru_cache

from src.core.config import load_yaml_config, settings
from src.core.exceptions import ConfigError, ModelProviderError
from src.core.types import ModelRole
from src.models.base import LLMProvider
from src.models.gemini_provider import GeminiProvider
from src.models.openrouter_provider import OpenRouterProvider

_PROVIDER_CLASSES = {
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
}


def _build_provider(provider_name: str, model_env_var: str) -> LLMProvider:
    # Resolve the model name from the validated `settings` object (which
    # already merges .env + real OS env vars + field defaults) rather than
    # os.getenv() directly — pydantic-settings parses .env into `settings`,
    # it does NOT export those values into the process environment, so
    # os.getenv() alone would miss anything only set in .env.
    model_name = getattr(settings, model_env_var.lower(), None) or os.getenv(model_env_var)
    if not model_name:
        raise ConfigError(
            f"Could not resolve a model for '{model_env_var}' (referenced from "
            f"configs/models.yaml). Set it in your .env file, e.g. {model_env_var}=gemini-3.6-flash"
        )

    provider_cls = _PROVIDER_CLASSES.get(provider_name)
    if provider_cls is None:
        raise ConfigError(f"Unknown provider '{provider_name}' in configs/models.yaml")

    return provider_cls(model=model_name)


@lru_cache
def _role_config() -> dict:
    return load_yaml_config("models.yaml")["roles"]


def get_provider_for_role(role: ModelRole | str) -> LLMProvider:
    role_name = role.value if isinstance(role, ModelRole) else role
    roles = _role_config()
    if role_name not in roles:
        raise ConfigError(f"No provider configured for role '{role_name}' in configs/models.yaml")

    cfg = roles[role_name]
    return _build_provider(cfg["provider"], cfg["model_env"])


@lru_cache
def _fallback_provider() -> LLMProvider:
    roles = _role_config()
    cfg = roles["fallback"]
    return _build_provider(cfg["provider"], cfg["model_env"])


def call_with_fallback(role: ModelRole | str, fn_name: str, *args, **kwargs):
    """Call `fn_name` (e.g. 'generate_json') on the primary provider for
    `role`; if it raises ModelProviderError, retry once against the
    globally configured fallback provider.

    If BOTH fail, the raised error includes the primary's failure reason
    too — not just the fallback's. Without this, a fallback failure (e.g.
    OpenRouter free-pool congestion) silently masks the real problem, which
    is usually why the primary failed in the first place."""
    primary = get_provider_for_role(role)
    try:
        return getattr(primary, fn_name)(*args, **kwargs)
    except ModelProviderError as primary_exc:
        fallback = _fallback_provider()
        try:
            return getattr(fallback, fn_name)(*args, **kwargs)
        except ModelProviderError as fallback_exc:
            raise ModelProviderError(
                f"Both providers failed for role '{role.value if isinstance(role, ModelRole) else role}'. "
                f"Primary ({primary.name}): {primary_exc} | "
                f"Fallback ({fallback.name}): {fallback_exc}"
            ) from fallback_exc
