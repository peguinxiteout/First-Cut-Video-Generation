"""Phase 1: image generation service, providers, and image agents.
No network: the Hugging Face client is replaced with a fake."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from src.agents.image import gap_analysis, generator
from src.agents.image.prompting import StyleProfile, build_prompt, derive_style, is_allowed
from src.data.schemas import ImageAnalysis, ImageBrief, PropertyInput, PropertyLocation
from src.models.image_gen import (
    ImageGenerationAuthError, ImageGenerationError, ImageRequest, ImageService, build_provider,
    get_image_service,
)
from src.models.image_gen.huggingface import HuggingFaceImageProvider
from src.models.image_gen.mock import MockImageProvider
from src.models.image_gen.profiles import fit_dimensions, profile_for

FLUX = "black-forest-labs/FLUX.1-schnell"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"


class _HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status) if status is not None else None


class FakeClient:
    """Stands in for huggingface_hub.InferenceClient. `script` is a list of
    outcomes per model: an int raises that HTTP status, None raises a
    transport error with no response, "ok" returns an image."""
    calls: list = []

    def __init__(self, model, token, timeout, script):
        self.model, self.script = model, list(script.get(model, ["ok"]))

    def text_to_image(self, **kwargs):
        FakeClient.calls.append((self.model, kwargs))
        outcome = self.script.pop(0) if self.script else "ok"
        if outcome == "ok":
            return Image.new("RGB", (kwargs["width"], kwargs["height"]), (120, 100, 80))
        if outcome == "none":
            raise _HTTPError(None)
        raise _HTTPError(outcome)


def _provider(script, models=(FLUX, SDXL)):
    FakeClient.calls = []
    return HuggingFaceImageProvider(
        token="hf_test", model_ids=list(models), max_retries=3,
        client_factory=lambda model, token, timeout: FakeClient(model, token, timeout, script),
        sleep=lambda s: None,
    )


# --- profiles -------------------------------------------------------------

def test_fit_dimensions_preserves_aspect_and_snaps_to_multiple():
    w, h = fit_dimensions(1600, 900, profile_for(FLUX))
    assert w % 16 == 0 and h % 16 == 0
    assert abs((w / h) - (16 / 9)) < 0.03
    assert w * h <= profile_for(FLUX).max_pixels * 1.02


def test_flux_gets_4_steps_and_no_negative_prompt(tmp_path):
    """Regression for the original test script, which sent FLUX.1-schnell
    30 steps plus a negative prompt it does not use."""
    provider = _provider({FLUX: ["ok"]})
    provider.generate(ImageRequest(prompt="p", negative_prompt="text"), tmp_path / "a.png")
    model, kwargs = FakeClient.calls[0]
    assert model == FLUX and kwargs["num_inference_steps"] == 4
    assert "negative_prompt" not in kwargs


def test_sdxl_receives_negative_prompt(tmp_path):
    provider = _provider({SDXL: ["ok"]}, models=(SDXL,))
    provider.generate(ImageRequest(prompt="p", negative_prompt="text"), tmp_path / "a.png")
    assert FakeClient.calls[0][1]["negative_prompt"] == "text"


# --- retry / fallback -----------------------------------------------------

def test_cold_start_503_is_retried_then_succeeds(tmp_path):
    result = _provider({FLUX: [503, 503, "ok"]}).generate(ImageRequest(prompt="p"), tmp_path / "a.png")
    assert result.model_id == FLUX and result.path.exists()


def test_unusable_model_falls_through_to_next(tmp_path):
    result = _provider({FLUX: [404], SDXL: ["ok"]}).generate(ImageRequest(prompt="p"), tmp_path / "a.png")
    assert result.model_id == SDXL


def test_bad_token_stops_immediately_without_trying_other_models(tmp_path):
    with pytest.raises(ImageGenerationAuthError):
        _provider({FLUX: [401], SDXL: ["ok"]}).generate(ImageRequest(prompt="p"), tmp_path / "a.png")
    assert {m for m, _ in FakeClient.calls} == {FLUX}


def test_error_without_response_object_does_not_crash(tmp_path):
    """The original read e.response.status_code unconditionally."""
    result = _provider({FLUX: ["none", "ok"]}).generate(ImageRequest(prompt="p"), tmp_path / "a.png")
    assert result.model_id == FLUX


def test_all_models_failing_raises_generation_error(tmp_path):
    with pytest.raises(ImageGenerationError):
        _provider({FLUX: [404], SDXL: [422]}).generate(ImageRequest(prompt="p"), tmp_path / "a.png")


# --- service / factory ----------------------------------------------------

def _settings(tmp_path, **overrides):
    base = dict(image_gen_provider="auto", hf_token="", image_gen_models=FLUX,
                cloudflare_account_id="", cloudflare_api_token="",
                cloudflare_image_models="@cf/black-forest-labs/flux-1-schnell",
                openrouter_api_key="", openrouter_image_models="black-forest-labs/flux-1.1-pro",
                image_gen_max_retries=2, image_gen_timeout_s=10, data_dir_resolved=tmp_path)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_missing_credentials_never_break_startup(tmp_path):
    """The original raised at IMPORT when HF_TOKEN was unset."""
    service = get_image_service(_settings(tmp_path))
    assert service.is_mock and service.enabled


def test_provider_none_disables_generation(tmp_path):
    service = ImageService(build_provider("none", _settings(tmp_path)), tmp_path)
    assert not service.enabled
    with pytest.raises(ImageGenerationError):
        service.generate(ImageRequest(prompt="p"))


def test_cache_generates_each_prompt_once(tmp_path):
    provider = _provider({FLUX: ["ok", "ok"]})
    service = ImageService(provider, tmp_path / "cache")
    first = service.generate(ImageRequest(prompt="same"), dest_path=tmp_path / "1.png")
    second = service.generate(ImageRequest(prompt="same"), dest_path=tmp_path / "2.png")
    assert not first.from_cache and second.from_cache
    assert len(FakeClient.calls) == 1


# --- prompting ------------------------------------------------------------

def _analysis(i, cat, caption=""):
    return ImageAnalysis(image_id=f"i{i}", category=cat, quality=7, composition=7, hero_score=5,
                         suggested_caption=caption)


def test_style_is_derived_from_the_real_photos():
    style = derive_style([_analysis(1, "living_room", "Warm oak flooring and brick walls at dusk"),
                          _analysis(2, "exterior", "Brick facade in warm evening light")])
    assert "brick" in style.materials and "warm" in style.palette


def test_prompt_avoids_ai_keyword_soup():
    prompt = build_prompt(ImageBrief(kind="lifestyle", purpose="p", subject="a terrace"),
                          StyleProfile(materials=["brick"], palette=["warm"]))
    for bad in ("8k", "masterpiece", "ultra-detailed"):
        assert bad not in prompt.lower()
    assert "brick" in prompt


@pytest.mark.parametrize("subject,ok", [
    ("abstract brick texture in raking light", True),
    ("a textured stone wall", True),
    ("the tower at dusk", False),
    ("site plan of the estate", False),
    ("floor plan with three levels", False),
])
def test_briefs_depicting_the_property_are_rejected(subject, ok):
    assert is_allowed(ImageBrief(kind="design_element", purpose="p", subject=subject)) is ok


# --- gap analysis ---------------------------------------------------------

def _prop(amenities=()):
    return PropertyInput(property_id="P", name="Sky Villas",
                         location=PropertyLocation(city="Noida", area="Sector 128"),
                         amenities=list(amenities))


def test_gap_analysis_filters_forbidden_and_duplicate_briefs():
    llm = lambda prompt: {"images": [
        {"kind": "lifestyle", "purpose": "p", "subject": "a family on a terrace"},
        {"kind": "lifestyle", "purpose": "p", "subject": "a family on a terrace"},    # duplicate
        {"kind": "design_element", "purpose": "p", "subject": "the tower facade at night"},  # forbidden
        {"kind": "exterior", "purpose": "p", "subject": "a villa"},                   # kind not allowed
        {"kind": "neighbourhood", "purpose": "p", "subject": "a boulevard at dusk"},
    ]}
    briefs = gap_analysis.run(_prop(), None, [_analysis(1, "living_room")], llm_call=llm)
    assert [b.kind for b in briefs] == ["lifestyle", "neighbourhood"]


def test_gap_analysis_always_returns_2_to_3_even_if_llm_fails():
    def broken(prompt):
        raise RuntimeError("quota exhausted")
    briefs = gap_analysis.run(_prop(["Swimming pool"]), None, [_analysis(1, "living_room")], llm_call=broken)
    assert 2 <= len(briefs) <= 3
    assert all(is_allowed(b) for b in briefs)


def test_gap_analysis_does_not_invent_a_photographed_amenity():
    photos = [_analysis(1, "amenity", "An infinity swimming pool with loungers")]
    briefs = gap_analysis.run(_prop(["Swimming pool"]), None, photos, llm_call=lambda p: {})
    assert not any(b.kind == "amenity_concept" and "pool" in b.subject for b in briefs)


# --- generator ------------------------------------------------------------

def test_generator_produces_labelled_visuals(tmp_path):
    service = ImageService(MockImageProvider(), tmp_path / "cache")
    briefs = [ImageBrief(kind="lifestyle", purpose="p", subject="a terrace", aspect="landscape"),
              ImageBrief(kind="neighbourhood", purpose="p", subject="a boulevard", aspect="wide")]
    report = generator.run(briefs, StyleProfile(), service, tmp_path / "out")
    assert len(report.visuals) == 2 and report.is_mock
    assert all(v.is_ai_generated for v in report.visuals)
    w, h = Image.open(report.visuals[1].path).size
    assert w > h  # wide brief -> wide image


def test_generator_stops_after_auth_error_and_keeps_going_after_soft_failure(tmp_path):
    # One scripted 404 per model per image: clients are cached per model,
    # so the script must cover both briefs.
    service = ImageService(_provider({FLUX: [404, 404], SDXL: [404, 404]}), tmp_path / "c")
    briefs = [ImageBrief(kind="lifestyle", purpose="p", subject="one"),
              ImageBrief(kind="lifestyle", purpose="p", subject="two")]
    report = generator.run(briefs, StyleProfile(), service, tmp_path / "o")
    assert report.visuals == [] and len(report.skipped) == 2  # both tried, both soft-failed

    service = ImageService(_provider({FLUX: [401]}), tmp_path / "c2")
    report = generator.run(briefs, StyleProfile(), service, tmp_path / "o2")
    assert len(report.skipped) == 1  # auth error: stop, don't hammer the API
