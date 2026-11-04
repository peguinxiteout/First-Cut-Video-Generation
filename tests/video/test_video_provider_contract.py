"""What every registered video adapter must satisfy, so a new adapter cannot
quietly go live: with no key it is unavailable, and constructing it touches
nothing (the socket guard in tests/conftest.py fails the test if it does)."""
from __future__ import annotations

import pytest

from src.models.video_gen import registry, selector
from src.models.video_gen.base import ProviderSpec, VideoProvider

FREE = {"kenburns", "mock"}


@pytest.fixture(scope="module")
def catalog():
    return selector.load_catalog()


@pytest.mark.parametrize("adapter", sorted(registry.ADAPTERS))
def test_every_registered_adapter_has_a_yaml_entry(adapter, catalog):
    assert [s for s in catalog.values() if s.adapter == adapter], f"no providers.yaml entry uses '{adapter}'"


@pytest.mark.parametrize("adapter", sorted(registry.ADAPTERS))
def test_with_no_key_only_a_provider_that_needs_none_is_available(adapter, catalog):
    for spec in (s for s in catalog.values() if s.adapter == adapter):
        provider = registry.ADAPTERS[adapter](spec, "")
        assert isinstance(provider, VideoProvider)
        assert provider.is_available() is (spec.key_env is None), spec.id


@pytest.mark.parametrize("adapter", sorted(registry.ADAPTERS))
def test_with_a_key_an_adapter_that_needs_one_is_available(adapter, catalog):
    for spec in (s for s in catalog.values() if s.adapter == adapter and s.key_env):
        assert registry.ADAPTERS[adapter](spec, "k").is_available()


def test_the_free_providers_are_always_available():
    for name in FREE:
        assert registry.ADAPTERS[name]().is_available()


def test_a_paid_looking_spec_on_any_adapter_is_unavailable_without_its_key():
    spec = ProviderSpec(id="x", label="X", adapter="mock", key_env="X_KEY")
    assert registry.ADAPTERS["mock"](spec, "").is_available() is False
    assert registry.ADAPTERS["mock"](spec, "  ").is_available() is False


def test_limits_and_prices_come_from_the_spec_not_the_class(catalog):
    kenburns = registry.ADAPTERS["kenburns"](catalog["kenburns"])
    assert kenburns.max_duration_s == catalog["kenburns"].max_duration_s
    assert kenburns.estimate_cost(15) == 0.0
    priced = registry.ADAPTERS["mock"](ProviderSpec(id="p", label="P", adapter="mock", cost_per_second=0.03))
    assert priced.estimate_cost(10) == 0.3
    unpriced = registry.ADAPTERS["mock"](ProviderSpec(id="u", label="U", adapter="mock", cost_per_second=None))
    assert unpriced.estimate_cost(10) is None


def test_every_provider_entry_names_an_adapter_and_the_fal_ones_are_not_installed_yet(catalog):
    for spec in catalog.values():
        assert spec.adapter
    fal = [s for s in catalog.values() if s.adapter == "fal"]
    assert {s.id for s in fal} == {"minimax_h3_max", "kling_25_turbo_pro", "seedance_2"}
    assert all(s.key_env == "FAL_KEY" for s in fal)
    assert "fal" not in registry.ADAPTERS
