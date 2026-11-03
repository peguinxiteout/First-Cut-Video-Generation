"""The image providers that replaced Hugging Face.

The Hugging Face path did not break — it ran out. Monthly credits hit
zero, every model returned 402 Payment Required, and the whole image
feature stopped at once, because one account's allowance was a single
point of failure that no amount of model-switching could route around.

So these tests are mostly about the failure modes rather than the happy
path: which errors mean "try the next model", which mean "try the next
VENDOR", and which mean "stop". Plus the two things Workers AI does that
nothing else does — a model with no width/height that silently returns a
square, and two response encodings behind the same 200 OK.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.models.image_gen.base import (
    ImageGenerationAuthError, ImageGenerationError, ImageGenerationQuotaError, ImageRequest,
)
from src.models.image_gen.chain import ChainImageProvider
from src.models.image_gen.cloudflare import CloudflareImageProvider
from src.models.image_gen.openrouter import (
    OpenRouterImageProvider, nearest_aspect_ratio, resolution_tier,
)

FLUX = "@cf/black-forest-labs/flux-1-schnell"
SDXL = "@cf/stabilityai/stable-diffusion-xl-base-1.0"
PNG = b"\x89PNG\r\n\x1a\n" + b"pretend-image"


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, content=b"", headers=None):
        self.status_code = status_code
        self._json = json_body
        self.content = content
        self.headers = headers or ({"content-type": "application/json"} if json_body else {})
        self.text = str(json_body or content[:40])

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json


def flux_ok():
    import base64
    return FakeResponse(json_body={"success": True,
                                   "result": {"image": base64.b64encode(PNG).decode()}})


def sdxl_ok():
    return FakeResponse(content=PNG, headers={"content-type": "image/png"})


def cf_error(status, *codes):
    return FakeResponse(status_code=status,
                        json_body={"success": False, "errors": [{"code": c} for c in codes]})


def cloudflare(responses, models=(FLUX, SDXL), **kwargs):
    """A provider whose transport replays `responses` per model id."""
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        model = url.split("/ai/run/", 1)[1]
        calls.append((model, json))
        queue = responses[model]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    provider = CloudflareImageProvider(account_id="acct", api_token="tok", model_ids=list(models),
                                       post=post, sleep=lambda _s: None, **kwargs)
    provider.calls = calls
    return provider


# --- Cloudflare: the two encodings -----------------------------------------

def test_a_base64_json_response_and_a_raw_byte_response_both_produce_an_image(tmp_path):
    """FLUX answers with base64 inside JSON, SDXL streams raw bytes, and
    both come back 200 OK. Assuming either one is how you get a PNG file
    full of JSON on the page."""
    for model, response in ((FLUX, flux_ok()), (SDXL, sdxl_ok())):
        out = tmp_path / f"{model[-8:]}.png"
        result = cloudflare({model: [response]}, models=(model,)).generate(
            ImageRequest(prompt="a villa", width=1024, height=1024), out)
        assert out.read_bytes() == PNG
        assert result.model_id == model


# --- Cloudflare: the square-only model --------------------------------------

def test_a_non_square_request_skips_the_model_that_cannot_honour_it(tmp_path):
    """flux-1-schnell has no width/height in its schema: asked for a 16:9
    hero it returns a square image and no error at all. That is the worst
    kind of failure — the page looks wrong and nothing says why — so the
    request goes to SDXL instead."""
    provider = cloudflare({SDXL: [sdxl_ok()]})
    provider.generate(ImageRequest(prompt="hero", width=1600, height=900), tmp_path / "a.png")
    assert [model for model, _ in provider.calls] == [SDXL]


def test_a_square_request_still_prefers_the_first_model(tmp_path):
    provider = cloudflare({FLUX: [flux_ok()]})
    provider.generate(ImageRequest(prompt="square", width=1024, height=1024), tmp_path / "a.png")
    assert [model for model, _ in provider.calls] == [FLUX]


def test_a_nearly_square_request_is_treated_as_square(tmp_path):
    provider = cloudflare({FLUX: [flux_ok()]})
    provider.generate(ImageRequest(prompt="nearly", width=1024, height=1000), tmp_path / "a.png")
    assert provider.calls[0][0] == FLUX


def test_the_square_only_model_is_not_sent_a_width_it_would_ignore(tmp_path):
    provider = cloudflare({FLUX: [flux_ok()]})
    provider.generate(ImageRequest(prompt="p", width=1024, height=1024), tmp_path / "a.png")
    _model, body = provider.calls[0]
    assert "width" not in body and "height" not in body
    assert body["steps"] <= 8, "Workers AI rejects flux above 8 steps"


def test_sdxl_is_kept_inside_the_step_limit_workers_ai_enforces(tmp_path):
    provider = cloudflare({SDXL: [sdxl_ok()]}, models=(SDXL,))
    provider.generate(ImageRequest(prompt="p", width=1600, height=900), tmp_path / "a.png")
    _model, body = provider.calls[0]
    assert body["num_steps"] <= 20
    assert 256 <= body["width"] <= 2048 and 256 <= body["height"] <= 2048


# --- Cloudflare: what each failure means ------------------------------------

def test_the_daily_allowance_running_out_is_a_quota_error_not_a_model_failure(tmp_path):
    """Error 3036 is this provider's version of the 402 that ended the
    Hugging Face path. Every model shares the same daily budget, so trying
    the next one is pointless — the only useful move is another vendor."""
    provider = cloudflare({FLUX: [cf_error(429, 3036)], SDXL: [sdxl_ok()]})
    with pytest.raises(ImageGenerationQuotaError):
        provider.generate(ImageRequest(prompt="p", width=1024, height=1024), tmp_path / "a.png")
    assert len(provider.calls) == 1, "no point asking a second model for the same budget"


def test_capacity_pressure_is_retried_rather_than_given_up_on(tmp_path):
    """3040 arrives on the same 429 as 3036 and means the opposite: busy
    now, fine in a moment."""
    provider = cloudflare({FLUX: [cf_error(429, 3040), flux_ok()]}, models=(FLUX,))
    provider.generate(ImageRequest(prompt="p", width=1024, height=1024), tmp_path / "a.png")
    assert len(provider.calls) == 2


def test_bad_credentials_stop_immediately(tmp_path):
    provider = cloudflare({FLUX: [FakeResponse(status_code=403)], SDXL: [sdxl_ok()]})
    with pytest.raises(ImageGenerationAuthError):
        provider.generate(ImageRequest(prompt="p", width=1024, height=1024), tmp_path / "a.png")
    assert len(provider.calls) == 1


def test_a_model_that_refuses_the_request_hands_over_to_the_next_one(tmp_path):
    provider = cloudflare({FLUX: [FakeResponse(status_code=400)], SDXL: [sdxl_ok()]})
    result = provider.generate(ImageRequest(prompt="p", width=1024, height=1024), tmp_path / "a.png")
    assert result.model_id == SDXL


def test_missing_configuration_is_reported_not_crashed(tmp_path):
    provider = CloudflareImageProvider(account_id="", api_token="", model_ids=[FLUX])
    assert not provider.is_available()
    with pytest.raises(ImageGenerationAuthError):
        provider.generate(ImageRequest(prompt="p"), tmp_path / "a.png")


# --- OpenRouter --------------------------------------------------------------

def openrouter(responses, models=("m1",)):
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        queue = responses
        return queue.pop(0) if len(queue) > 1 else queue[0]

    provider = OpenRouterImageProvider(api_key="k", model_ids=list(models), post=post,
                                       sleep=lambda _s: None)
    provider.calls = calls
    return provider


def or_ok():
    import base64
    return FakeResponse(json_body={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})


def test_openrouter_returns_an_image(tmp_path):
    out = tmp_path / "a.png"
    openrouter([or_ok()]).generate(ImageRequest(prompt="p", width=1600, height=900), out)
    assert out.read_bytes() == PNG


def test_insufficient_credit_is_a_quota_error(tmp_path):
    """402 here is the same condition that ended Hugging Face. It is not a
    transient error and must not be retried."""
    provider = openrouter([FakeResponse(status_code=402)])
    with pytest.raises(ImageGenerationQuotaError):
        provider.generate(ImageRequest(prompt="p"), tmp_path / "a.png")
    assert len(provider.calls) == 1


def test_a_moderation_block_is_not_retried(tmp_path):
    """Rewording might pass; asking the same thing again will not."""
    provider = openrouter([FakeResponse(status_code=403)])
    with pytest.raises(ImageGenerationError):
        provider.generate(ImageRequest(prompt="p"), tmp_path / "a.png")
    assert len(provider.calls) == 1


@pytest.mark.parametrize("width,height,expected", [
    (1600, 900, "16:9"), (1024, 1024, "1:1"), (900, 1600, "9:16"),
    (1200, 900, "4:3"), (900, 1200, "3:4"), (1500, 1000, "3:2"),
])
def test_a_requested_size_becomes_the_nearest_supported_aspect_ratio(width, height, expected):
    assert nearest_aspect_ratio(width, height) == expected


def test_portrait_never_comes_back_landscape():
    """Compared in log space on purpose: in linear space the wide side
    dominates and a 2:3 request rounds to a landscape ratio."""
    for width, height in ((800, 1200), (600, 1400), (1000, 1100)):
        assert nearest_aspect_ratio(width, height).startswith(("9:", "2:", "3:", "1:"))
        ratio_w, ratio_h = (int(n) for n in nearest_aspect_ratio(width, height).split(":"))
        assert ratio_w <= ratio_h


def test_resolution_tier_tracks_the_requested_size():
    assert resolution_tier(640, 480) == "512"
    assert resolution_tier(1600, 900) == "2K"


# --- the chain ---------------------------------------------------------------

class Stub:
    def __init__(self, name, error=None, available=True):
        self.name, self._error, self._available = name, error, available
        self.calls = 0

    def is_available(self):
        return self._available

    def generate(self, request, output_path):
        self.calls += 1
        if self._error:
            raise self._error
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(PNG)
        from src.models.image_gen.base import GeneratedImage
        return GeneratedImage(path=output_path, model_id=self.name, prompt=request.prompt,
                              width=request.width, height=request.height)


def test_an_exhausted_vendor_hands_over_to_the_next_one(tmp_path):
    """The whole point. When Hugging Face ran out there was nowhere to go,
    because the only fallback was another model on the same account."""
    primary = Stub("cloudflare", ImageGenerationQuotaError("daily allowance spent"))
    backup = Stub("openrouter")
    result = ChainImageProvider([primary, backup]).generate(
        ImageRequest(prompt="p"), tmp_path / "a.png")
    assert result.model_id == "openrouter"
    assert primary.calls == 1 and backup.calls == 1


def test_a_vendor_that_is_not_configured_is_skipped_without_an_error(tmp_path):
    primary = Stub("cloudflare", available=False)
    backup = Stub("openrouter")
    ChainImageProvider([primary, backup]).generate(ImageRequest(prompt="p"), tmp_path / "a.png")
    assert primary.calls == 0 and backup.calls == 1


def test_when_every_vendor_fails_the_reasons_are_all_reported(tmp_path):
    """One line naming both vendors, so the person can see whether they are
    out of allowance or out of credentials without reading the logs."""
    chain = ChainImageProvider([
        Stub("cloudflare", ImageGenerationQuotaError("daily allowance spent")),
        Stub("openrouter", ImageGenerationQuotaError("needs $1 of credit")),
    ])
    with pytest.raises(ImageGenerationError) as caught:
        chain.generate(ImageRequest(prompt="p"), tmp_path / "a.png")
    assert "cloudflare" in str(caught.value) and "openrouter" in str(caught.value)


def test_the_cache_key_covers_every_model_the_chain_could_use(tmp_path):
    """The service keys its cache on the provider's models. If a chain
    reported none, changing providers would keep serving images the old
    one made."""
    chain = ChainImageProvider([
        CloudflareImageProvider("a", "b", [FLUX]),
        OpenRouterImageProvider("k", ["flux-pro"]),
    ])
    assert set(chain._model_ids) == {FLUX, "flux-pro"}


# --- the factory -------------------------------------------------------------

def _settings(**overrides):
    base = dict(image_gen_provider="auto", hf_token="tok-that-is-out-of-credit",
                image_gen_models="black-forest-labs/FLUX.1-schnell",
                cloudflare_account_id="", cloudflare_api_token="",
                cloudflare_image_models=f"{FLUX},{SDXL}",
                openrouter_api_key="", openrouter_image_models="black-forest-labs/flux-1.1-pro",
                image_gen_max_retries=2, image_gen_timeout_s=10)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_hugging_face_is_never_chosen_automatically_even_with_a_token():
    """It is not that the token is wrong — the ACCOUNT is out of credit, so
    including it in the automatic chain would add a 402 and a delay to
    every request. It stays selectable by name for anyone who buys more."""
    from src.models.image_gen import build_provider

    provider = build_provider("auto", _settings(cloudflare_account_id="a", cloudflare_api_token="b"))
    assert provider.name == "cloudflare"
    assert build_provider("huggingface", _settings()).name == "huggingface"


def test_the_free_provider_is_tried_before_the_paid_one():
    from src.models.image_gen import build_provider

    provider = build_provider("auto", _settings(cloudflare_account_id="a", cloudflare_api_token="b",
                                                openrouter_api_key="k"))
    assert [p.name for p in provider.providers] == ["cloudflare", "openrouter"]


def test_with_nothing_configured_the_pipelines_still_run():
    from src.models.image_gen import build_provider

    assert build_provider("auto", _settings()).name == "mock"
