"""A video run reaches Gemini only: other vendors' keys are blanked, a fallback to one is refused and the record says so."""
from __future__ import annotations

import os

import pytest

from src.agents.video import offline, policy
from src.core.config import settings
from src.core.exceptions import ModelProviderError
from src.core.types import ModelRole
from src.models import gateway
from src.workflows import video as workflow
from tests.video.helpers import make_video_property


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("src.agents.base.time.sleep", lambda s: None)


class FakeProvider:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, 0

    def generate_json(self, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise ModelProviderError(f"{self.name} is down")
        return {"ok": self.name}

    generate_json_with_images = generate_json


@pytest.fixture
def providers(monkeypatch):
    primary, fallback = FakeProvider("gemini", fail=True), FakeProvider("openrouter")
    monkeypatch.setattr(gateway, "get_provider_for_role", lambda role: primary)
    monkeypatch.setattr(gateway, "_fallback_provider", lambda: fallback)
    return primary, fallback


# -- the gateway guard --------------------------------------------------------------------------------------------

def test_without_a_policy_a_failed_gemini_call_falls_back_to_openrouter(providers):
    primary, fallback = providers
    assert gateway.call_with_fallback(ModelRole.VISION, "generate_json", "p") == {"ok": "openrouter"}
    assert (primary.calls, fallback.calls) == (1, 1)


def test_under_a_gemini_only_policy_the_fallback_is_refused_and_listed(providers):
    primary, fallback = providers
    with gateway.restricted(vendors=["gemini"]) as call_policy:
        with pytest.raises(ModelProviderError, match="gemini is down"):                 # the primary's own failure, not a new one
            gateway.call_with_fallback(ModelRole.VISION, "generate_json", "p")
    assert (primary.calls, fallback.calls) == (1, 0)
    assert call_policy.blocked == ["fallback to openrouter for role 'vision'"]


def test_a_role_that_is_not_allowed_is_refused_before_any_call(providers):
    primary, fallback = providers
    with gateway.restricted(vendors=["gemini"], roles=["vision"]) as call_policy:
        with pytest.raises(ModelProviderError, match="blocked by the run's call policy"):
            gateway.call_with_fallback(ModelRole.COPYWRITER, "generate_json", "p")
    assert primary.calls == fallback.calls == 0
    assert call_policy.blocked == ["role 'copywriter' may not be asked in this run"]


def test_a_role_that_is_listed_is_allowed_and_its_success_blocks_nothing(monkeypatch):
    ok = FakeProvider("gemini")
    monkeypatch.setattr(gateway, "get_provider_for_role", lambda role: ok)
    with gateway.restricted(vendors=["gemini"], roles=["vision"]) as call_policy:
        assert gateway.call_with_fallback(ModelRole.VISION, "generate_json", "p") == {"ok": "gemini"}
    assert call_policy.blocked == []


def test_the_policy_ends_with_the_block(providers):
    with gateway.restricted(vendors=["gemini"]):
        pass
    assert gateway.call_with_fallback(ModelRole.VISION, "generate_json", "p") == {"ok": "openrouter"}


def test_an_empty_policy_limits_nothing(providers):
    with gateway.restricted():
        assert gateway.call_with_fallback(ModelRole.VISION, "generate_json", "p") == {"ok": "openrouter"}


# -- the run policy ---------------------------------------------------------------------------------------------------

def test_the_shipped_policy_is_gemini_cloud_tts_and_beatoven_with_the_gemini_roles():
    from src.core.config import load_yaml_config

    config = load_yaml_config("video/policy.yaml")
    roles = load_yaml_config("models.yaml")["roles"]
    assert config["allowed_vendors"] == ["gemini", "google_cloud_tts", "beatoven"]
    assert not {"openrouter", "cloudflare", "fal", "huggingface"} & set(config["allowed_vendors"])
    assert {"vision", "property_intelligence"} <= set(config["model_roles"])
    assert all(roles[r]["provider"] == "gemini" for r in config["model_roles"])          # no role of another vendor
    assert "fallback" not in config["model_roles"]                                       # the OpenRouter fallback is not a role to allow


def test_other_vendors_keys_are_blanked_for_the_run_and_restored_after(monkeypatch):
    for name, value in (("gemini_api_key", "g"), ("openrouter_api_key", "o"), ("cloudflare_account_id", "c1"),
                        ("cloudflare_api_token", "c2"), ("hf_token", "h"), ("tavily_api_key", "t"),
                        ("google_cloud_tts_api_key", "v"), ("beatoven_api_key", "m")):
        monkeypatch.setattr(settings, name, value, raising=False)
    monkeypatch.setattr(settings, "tools_enabled", True, raising=False)
    monkeypatch.setattr(settings, "image_gen_provider", "auto", raising=False)
    monkeypatch.setattr(settings, "video_gen_provider", "auto", raising=False)
    monkeypatch.setenv("FAL_KEY", "f")

    with policy.video_run_policy():
        assert (settings.gemini_api_key, settings.google_cloud_tts_api_key, settings.beatoven_api_key) == ("g", "v", "m")
        assert not any((settings.openrouter_api_key, settings.cloudflare_account_id, settings.cloudflare_api_token,
                        settings.hf_token, settings.tavily_api_key))
        assert settings.tools_enabled is False and settings.image_gen_provider == "mock"
        assert settings.video_gen_provider == "kenburns" and "FAL_KEY" not in os.environ

    assert (settings.openrouter_api_key, settings.cloudflare_api_token, settings.hf_token) == ("o", "c2", "h")
    assert settings.tools_enabled is True and settings.image_gen_provider == "auto" and os.environ["FAL_KEY"] == "f"


def test_a_policy_that_lists_only_gemini_blanks_the_voice_and_music_keys(monkeypatch):
    monkeypatch.setattr(settings, "google_cloud_tts_api_key", "v", raising=False)
    monkeypatch.setattr(settings, "beatoven_api_key", "m", raising=False)
    monkeypatch.setattr(policy, "_config", lambda: {"allowed_vendors": ["gemini"], "model_roles": []})
    with policy.video_run_policy():
        assert (settings.google_cloud_tts_api_key, settings.beatoven_api_key) == ("", "")
    assert (settings.google_cloud_tts_api_key, settings.beatoven_api_key) == ("v", "m")


def test_the_keys_come_back_even_when_the_run_fails(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "o", raising=False)
    with pytest.raises(RuntimeError):
        with policy.video_run_policy():
            raise RuntimeError("boom")
    assert settings.openrouter_api_key == "o" and policy.active() is None


def test_a_role_outside_the_policy_takes_its_own_fallback_without_asking(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "g", raising=False)
    with policy.video_run_policy():
        assert offline.model_blocker(False, "vision") == ""
        assert offline.model_blocker(False, "property_intelligence") == ""                  # a Gemini text role is allowed
        assert "does not allow the 'blog_writer' model" in offline.model_blocker(False, "blog_writer")
    assert offline.model_blocker(False, "blog_writer") == ""                                 # no run, no policy
    assert offline.model_blocker(True, "vision") == "offline run"


def test_a_policy_with_nothing_listed_changes_nothing(monkeypatch):
    monkeypatch.setattr(policy, "_config", lambda: {"allowed_vendors": [], "model_roles": []})
    monkeypatch.setattr(settings, "openrouter_api_key", "o", raising=False)
    with policy.video_run_policy() as call_policy:
        assert call_policy is None and settings.openrouter_api_key == "o"


# -- in a run ----------------------------------------------------------------------------------------------------------------

@pytest.fixture
def run_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "v.db"), raising=False)
    monkeypatch.setattr(settings, "gemini_api_key", "g", raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "o", raising=False)
    return make_video_property(tmp_path / "properties")


def test_a_run_whose_gemini_vision_fails_does_not_reach_openrouter_and_the_record_says_so(run_environment, providers):
    primary, fallback = providers
    record = workflow.run_plan_workflow(run_environment, draft=True)

    assert record.status == "planned"
    assert fallback.calls == 0 and primary.calls >= 1
    assert "Blocked by the run's policy: fallback to openrouter for role 'vision'." in record.notes
    assert any("read from the uploads (vision unavailable" in n for n in record.notes)           # the offline reading took over
    assert "Blocked by the run's policy: fallback to openrouter for role 'property_intelligence'." in record.notes
    assert record.agent_reports["fact_ledger"]                                                  # the ledger ran, from the form


def test_a_run_with_no_failures_blocks_nothing(run_environment, monkeypatch):
    ok = FakeProvider("gemini")
    monkeypatch.setattr(gateway, "get_provider_for_role", lambda role: ok)
    monkeypatch.setattr("src.agents.video.vision_cache.classify_batch",
                        lambda images: (_ for _ in ()).throw(ModelProviderError("x")))
    record = workflow.run_plan_workflow(run_environment, draft=True)
    assert not any(n.startswith("Blocked by the run's policy") for n in record.notes)
