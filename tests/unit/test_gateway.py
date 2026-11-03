from __future__ import annotations

from src.core.config import settings
from src.models import gateway


def test_build_provider_resolves_model_from_settings_not_raw_os_environ(monkeypatch):
    """Regression test: the model gateway must resolve model names from the
    validated `settings` object, not os.getenv() directly. pydantic-settings
    parses .env into `settings` but does NOT export those values into the
    real process environment, so relying on os.getenv() alone silently
    breaks model resolution even when .env is filled in correctly."""
    # Simulate a fresh shell where the env var was never explicitly exported
    # (the normal case for anyone just editing .env).
    monkeypatch.delenv("GEMINI_TEXT_MODEL", raising=False)

    captured = {}

    class DummyProvider:
        def __init__(self, model):
            captured["model"] = model

    monkeypatch.setitem(gateway._PROVIDER_CLASSES, "gemini", DummyProvider)

    provider = gateway._build_provider("gemini", "GEMINI_TEXT_MODEL")

    assert captured["model"] == settings.gemini_text_model
    assert isinstance(provider, DummyProvider)


def test_build_provider_raises_config_error_when_truly_unresolvable(monkeypatch):
    monkeypatch.delenv("NOT_A_REAL_SETTING", raising=False)
    from src.core.exceptions import ConfigError
    import pytest

    with pytest.raises(ConfigError):
        gateway._build_provider("gemini", "NOT_A_REAL_SETTING")


def test_call_with_fallback_surfaces_both_errors_when_both_providers_fail(monkeypatch):
    """Regression test: when both the primary and fallback providers fail,
    the raised error must mention BOTH failure reasons. Swallowing the
    primary's error (as the old code did) hides the actual root cause —
    e.g. a Gemini quota error that's driving every call to the already-
    congested OpenRouter free pool looks, from the outside, identical to
    "OpenRouter is just rate-limited" even though the real fix is on the
    Gemini side."""
    import pytest
    from src.core.exceptions import ModelProviderError

    class FailingPrimary:
        name = "gemini"

        def generate_json(self, *a, **k):
            raise ModelProviderError("Gemini: daily quota exceeded")

    class FailingFallback:
        name = "openrouter"

        def generate_json(self, *a, **k):
            raise ModelProviderError("OpenRouter: 429 rate-limited upstream")

    monkeypatch.setattr(gateway, "get_provider_for_role", lambda role: FailingPrimary())
    monkeypatch.setattr(gateway, "_fallback_provider", lambda: FailingFallback())

    with pytest.raises(ModelProviderError) as exc_info:
        gateway.call_with_fallback("copywriter", "generate_json", "some prompt")

    message = str(exc_info.value)
    assert "daily quota exceeded" in message
    assert "429 rate-limited upstream" in message


def test_call_with_fallback_succeeds_via_fallback_when_only_primary_fails(monkeypatch):
    from src.core.exceptions import ModelProviderError

    class FailingPrimary:
        name = "gemini"

        def generate_json(self, *a, **k):
            raise ModelProviderError("transient error")

    class WorkingFallback:
        name = "openrouter"

        def generate_json(self, *a, **k):
            return {"ok": True}

    monkeypatch.setattr(gateway, "get_provider_for_role", lambda role: FailingPrimary())
    monkeypatch.setattr(gateway, "_fallback_provider", lambda: WorkingFallback())

    result = gateway.call_with_fallback("copywriter", "generate_json", "some prompt")
    assert result == {"ok": True}


def test_vision_and_critic_roles_resolve_to_different_models():
    """Regression test for a real production bug: vision and critic both
    pointed at GEMINI_VISION_MODEL, silently sharing one 20-requests/day
    free-tier quota bucket between two different agents — a generation
    would see vision succeed and then critic fail with a 429, even though
    the text/fast quota pools sat completely untouched. They must resolve
    to genuinely different model names now."""
    from src.core.config import load_yaml_config

    roles_cfg = load_yaml_config("models.yaml")["roles"]
    vision_model_env = roles_cfg["vision"]["model_env"]
    critic_model_env = roles_cfg["critic"]["model_env"]

    assert vision_model_env != critic_model_env

    vision_model = getattr(settings, vision_model_env.lower())
    critic_model = getattr(settings, critic_model_env.lower())
    assert vision_model != critic_model


def test_blog_critic_role_does_not_share_the_vision_model_either():
    from src.core.config import load_yaml_config

    roles_cfg = load_yaml_config("models.yaml")["roles"]
    assert roles_cfg["blog_critic"]["model_env"] != roles_cfg["vision"]["model_env"]


def test_openrouter_fallback_uses_the_free_models_router_not_a_pinned_slug():
    """Regression test: a pinned free-model slug (e.g.
    'minimax/minimax-m3:free') can be discontinued by OpenRouter without
    warning — which is exactly what happened and prompted this test.
    openrouter/free is OpenRouter's own maintained router and doesn't
    have this failure mode."""
    assert settings.openrouter_model == "openrouter/free"
