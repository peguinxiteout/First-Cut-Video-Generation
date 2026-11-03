from __future__ import annotations

import pytest

from src.data.schemas import PropertyContact, PropertyInput, PropertyLocation


@pytest.fixture
def sample_property_input() -> PropertyInput:
    return PropertyInput(
        property_id="PROP-TEST",
        name="Test Residences",
        developer="Test Developer",
        location=PropertyLocation(city="Ahmedabad", area="SG Highway"),
        configuration="3 BHK",
        area="1450 sq ft",
        price="₹1.2 Cr",
        amenities=["Swimming Pool", "Gym", "Clubhouse"],
        description="A test property used only in unit tests.",
        contact=PropertyContact(phone="+91-00000-00000", email="sales@example.com"),
    )


@pytest.fixture(autouse=True)
def _offline_image_generation(tmp_path, monkeypatch):
    """Every test runs image generation OFFLINE, into a temp directory.

    Without this, a developer with real credentials in .env would have the
    integration tests call a live image API — slow, flaky, and spending a
    real allowance — and every run would leave generated files in
    data/image_cache. Tests that exercise a provider inject a fake
    transport directly (see tests/unit/test_image_gen.py), so nothing here
    needs the network.
    """
    from src.core.config import settings

    monkeypatch.setattr(settings, "image_gen_provider", "mock", raising=False)
    monkeypatch.setattr(settings, "hf_token", "", raising=False)
    monkeypatch.setattr(settings, "cloudflare_account_id", "", raising=False)
    monkeypatch.setattr(settings, "cloudflare_api_token", "", raising=False)
    # And no research tool reaches the network either. Same reasoning: a
    # test that quietly called OpenStreetMap or Google News would be slow,
    # flaky, dependent on somebody else's uptime, and — for the APIs that
    # ask you not to hammer them — rude. Tests that exercise a tool inject
    # a fake transport (see tests/unit/test_tools.py).
    monkeypatch.setattr(settings, "tools_enabled", False, raising=False)
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"), raising=False)
