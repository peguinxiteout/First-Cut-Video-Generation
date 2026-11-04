from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.core.exceptions import ConfigError
from src.models.video_gen import registry, selector
from src.models.video_gen.mock import MockVideoProvider


class FakeFal(MockVideoProvider):
    """Stands in for the fal adapter, which arrives in Phase 3."""


@pytest.fixture
def fal_installed(monkeypatch):
    monkeypatch.setitem(registry.ADAPTERS, "fal", FakeFal)


def _by_id(duration=15, **kwargs):
    return {o.id: o for o in selector.options(duration, **kwargs)}


def test_the_list_is_auto_then_each_provider_then_pan_and_zoom():
    ids = [o.id for o in selector.options(15)]
    assert ids == ["auto", "minimax_h3_max", "kling_25_turbo_pro", "seedance_2", "kenburns"]
    assert "mock" not in ids                       # hidden


def test_before_the_fal_adapter_exists_the_paid_models_are_listed_as_unavailable():
    options = _by_id()
    for name in ("minimax_h3_max", "kling_25_turbo_pro", "seedance_2"):
        assert options[name].available is False
        assert options[name].reason == "adapter not installed"
    assert options["kenburns"].available and options["kenburns"].label == "Pan-and-zoom (free)"
    assert options["auto"].available
    assert options["auto"].reason == "No video model is available; uses pan-and-zoom"


def test_the_estimate_is_the_rate_times_the_duration_and_none_when_no_price_is_set():
    options = _by_id(15)
    assert options["minimax_h3_max"].est_cost_usd == pytest.approx(0.45)
    assert options["kling_25_turbo_pro"].est_cost_usd == pytest.approx(1.05)
    assert options["seedance_2"].est_cost_usd is None
    assert options["kenburns"].est_cost_usd == 0.0
    assert _by_id(10)["minimax_h3_max"].est_cost_usd == pytest.approx(0.30)


def test_an_installed_adapter_without_a_key_says_which_key(fal_installed):
    options = _by_id()
    assert options["minimax_h3_max"].available is False
    assert options["minimax_h3_max"].reason == "no key: set FAL_KEY"


def test_with_a_key_the_paid_models_are_available_and_auto_takes_the_lowest_priority(fal_installed, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    options = _by_id(15)
    assert options["minimax_h3_max"].available and options["kling_25_turbo_pro"].available
    assert options["auto"].reason == "Uses MiniMax H3 Max"
    assert options["auto"].est_cost_usd == pytest.approx(0.45)
    assert selector.resolve("auto", 15).ids == ("minimax_h3_max", "kenburns")


def test_auto_never_picks_a_provider_without_a_priority(fal_installed, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    assert "seedance_2" not in selector.resolve("auto").ids
    assert selector.resolve("seedance_2").ids == ("seedance_2", "kenburns")     # but it can be chosen


def test_a_key_in_settings_counts_as_well_as_one_in_the_environment(fal_installed, tmp_path):
    # Settings has no fal_key field until Phase 3; a stub stands in for it.
    stub = SimpleNamespace(fal_key="from-settings", data_dir_resolved=tmp_path, video_gen_provider="auto")
    assert _by_id(settings=stub)["minimax_h3_max"].available
    assert selector.resolve("auto", settings=stub).ids == ("minimax_h3_max", "kenburns")
    assert selector.resolve_key("FAL_KEY", SimpleNamespace(fal_key="  ")) == ""


def test_a_spent_allowance_removes_the_provider_until_the_cooldown_ends(fal_installed, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    now = datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)
    until = selector.mark_quota("minimax_h3_max", now=now)
    assert until == now + timedelta(hours=12)

    options = _by_id(now=now)
    assert options["minimax_h3_max"].available is False
    assert options["minimax_h3_max"].reason == "allowance spent; skipped until 21:00 UTC"
    assert selector.resolve("auto", now=now).ids == ("kling_25_turbo_pro", "kenburns")

    later = now + timedelta(hours=13)
    assert _by_id(now=later)["minimax_h3_max"].available
    assert selector.resolve("auto", now=later).ids[0] == "minimax_h3_max"


def test_a_garbled_marker_file_is_ignored(fal_installed, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    path = selector._marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert _by_id()["minimax_h3_max"].available


class TestChains:
    def test_auto_with_nothing_available_is_pan_and_zoom_alone(self):
        plan = selector.resolve("auto", 15)
        assert plan.ids == ("kenburns",) and plan.est_cost_usd == 0.0 and plan.note == ""

    def test_a_named_provider_is_followed_by_pan_and_zoom(self, fal_installed, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "k")
        plan = selector.resolve("kling_25_turbo_pro", 15)
        assert plan.ids == ("kling_25_turbo_pro", "kenburns")
        assert plan.est_cost_usd == pytest.approx(1.05)

    def test_pan_and_zoom_never_calls_a_model(self, fal_installed, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "k")
        assert selector.resolve("kenburns").ids == ("kenburns",)

    def test_a_named_provider_that_cannot_run_gives_pan_and_zoom_and_says_why(self):
        plan = selector.resolve("minimax_h3_max", 15)
        assert plan.ids == ("kenburns",)
        assert plan.note == "MiniMax H3 Max is unavailable (adapter not installed); using pan-and-zoom"

    def test_the_choice_is_case_insensitive_and_defaults_to_the_setting(self, monkeypatch):
        assert selector.resolve("  KenBurns ").ids == ("kenburns",)
        monkeypatch.setattr("src.core.config.settings.video_gen_provider", "kenburns")
        assert selector.resolve(None).ids == ("kenburns",)
        assert selector.resolve("").ids == ("kenburns",)

    def test_an_unknown_provider_is_a_config_error_naming_the_choices(self):
        with pytest.raises(ConfigError, match="minimax_h3_max"):
            selector.resolve("sora")
