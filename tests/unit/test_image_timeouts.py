"""A slow image vendor must cost an illustration, never a brochure.

The failure these cover, verbatim from a real run:

    Generation failed: HTTPSConnectionPool(host='api.cloudflare.com',
    port=443): Read timed out. (read timeout=120.0)
    1 step(s) failed: visual_enrichment

Nothing about a 120-second image request is unusual. What made it fatal is
that `requests.exceptions.ReadTimeout` is not an `ImageGenerationError`, so
it walked past every layer built to absorb a failed image: the provider's
retry loop, the next model, the failover to the second vendor, and the
agent's soft failure. Each of those is a separate test here, because each
was separately broken by the same gap.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.models.image_gen import transport
from src.models.image_gen.base import (
    ImageGenerationError, ImageGenerationQuotaError, ImageGenerationTransportError, ImageRequest,
)
from src.models.image_gen.chain import ChainImageProvider
from src.models.image_gen.cloudflare import CloudflareImageProvider
from src.models.image_gen.openrouter import OpenRouterImageProvider


class ReadTimeout(Exception):
    """Stands in for requests.exceptions.ReadTimeout.

    Deliberately NOT a subclass of anything this project defines — that is
    the whole point. The real one comes from `requests`, knows nothing
    about us, and the fix has to work for an exception type nobody here
    anticipated."""


class _Response:
    def __init__(self, status_code=200, content=b"PNG", headers=None, text=""):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"content-type": "image/png"}
        self.text = text

    def json(self):
        return {}


def _request() -> ImageRequest:
    return ImageRequest(prompt="a courtyard at dusk", width=1024, height=1024)


def _cloudflare(post, **kwargs):
    return CloudflareImageProvider(
        account_id="acct", api_token="tok",
        model_ids=["@cf/stabilityai/stable-diffusion-xl-base-1.0"],
        post=post, sleep=lambda _s: None, **kwargs)


# --------------------------------------------------------------------------
# 1. the retry loop has to SEE it
# --------------------------------------------------------------------------

def test_a_read_timeout_is_retried_rather_than_raised_at_the_caller(tmp_path):
    calls = {"n": 0}

    def post(*_a, **_k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ReadTimeout("HTTPSConnectionPool(host='api.cloudflare.com', port=443): "
                              "Read timed out. (read timeout=120.0)")
        return _Response()

    image = _cloudflare(post).generate(_request(), tmp_path / "out.png")
    assert calls["n"] == 3, "the timeout must be retried, not raised on the first attempt"
    assert image.path.exists()


def test_a_timeout_that_never_clears_becomes_a_soft_failure(tmp_path):
    def post(*_a, **_k):
        raise ReadTimeout("Read timed out. (read timeout=120.0)")

    with pytest.raises(ImageGenerationError) as caught:
        _cloudflare(post).generate(_request(), tmp_path / "out.png")
    # ImageGenerationError is what every caller treats as "carry on without
    # the image". If this ever stops being one, the brochure dies again.
    assert isinstance(caught.value, ImageGenerationError)
    assert "did not answer in time" in str(caught.value), (
        "the message has to tell an operator what happened, not quote a connection pool")


def test_the_same_hole_is_closed_in_the_other_vendor(tmp_path):
    def post(*_a, **_k):
        raise ReadTimeout("Read timed out.")

    provider = OpenRouterImageProvider(api_key="k", model_ids=["some/model"],
                                       post=post, sleep=lambda _s: None)
    with pytest.raises(ImageGenerationError):
        provider.generate(_request(), tmp_path / "out.png")


# --------------------------------------------------------------------------
# 2. the chain has to fail OVER on it
# --------------------------------------------------------------------------

def test_a_vendor_that_times_out_hands_over_to_the_other_one(tmp_path):
    """The reason the chain exists. A timeout used to escape it entirely,
    so the second vendor — configured, funded and working — was never
    asked."""
    def times_out(*_a, **_k):
        raise ReadTimeout("Read timed out. (read timeout=120.0)")

    served = {"n": 0}

    def serves(*_a, **_k):
        served["n"] += 1
        return _Response()

    chain = ChainImageProvider([
        _cloudflare(times_out),
        CloudflareImageProvider(account_id="b", api_token="t2", model_ids=["@cf/other/model"],
                                post=serves, sleep=lambda _s: None),
    ])
    image = chain.generate(_request(), tmp_path / "out.png")
    assert served["n"] == 1, "the second vendor was never asked"
    assert image.path.exists()


# --------------------------------------------------------------------------
# 3. the whole attempt has to be BOUNDED
# --------------------------------------------------------------------------

def test_one_providers_attempt_is_bounded_in_time_not_in_retries(tmp_path):
    """Four retries across two models at a 120s read timeout is sixteen
    minutes of a progress bar going nowhere. The budget caps the provider,
    so a slow vendor costs a bounded wait and the chain moves on."""
    clock = {"t": 0.0}
    original = transport.time.monotonic
    transport.time.monotonic = lambda: clock["t"]          # type: ignore[assignment]
    try:
        def post(*_a, **_k):
            clock["t"] += 90.0                              # each attempt is slow
            raise ReadTimeout("Read timed out.")

        provider = CloudflareImageProvider(
            account_id="a", api_token="t",
            model_ids=["@cf/one/model", "@cf/two/model", "@cf/three/model"],
            max_retries=4, post=post, sleep=lambda _s: None)
        with pytest.raises(ImageGenerationError):
            provider.generate(_request(), tmp_path / "out.png")
    finally:
        transport.time.monotonic = original                 # type: ignore[assignment]
    assert clock["t"] <= transport.DEFAULT_BUDGET_S + 90.0, (
        f"spent {clock['t']:.0f}s against a {transport.DEFAULT_BUDGET_S:.0f}s budget")


def test_the_read_timeout_shrinks_to_what_is_left_of_the_budget():
    """Otherwise the last attempt inside a budget still blocks for the full
    read timeout, and a 210s budget quietly becomes 330s."""
    clock = {"t": 0.0}
    original = transport.time.monotonic
    transport.time.monotonic = lambda: clock["t"]            # type: ignore[assignment]
    try:
        at = transport.deadline(100.0)
        assert transport.remaining(at, 120.0) == pytest.approx(100.0)
        clock["t"] = 80.0
        assert transport.remaining(at, 120.0) == pytest.approx(20.0)
    finally:
        transport.time.monotonic = original                  # type: ignore[assignment]


def test_connect_and_read_timeouts_are_separate_numbers():
    """One number for both means a vendor that is DOWN is waited on for as
    long as a vendor that is generating."""
    seen = {}

    def post(url, **kwargs):
        seen.update(kwargs)
        return _Response()

    transport.call(post, "https://example.invalid", headers={}, json={}, read_timeout=120.0)
    assert isinstance(seen["timeout"], tuple), "timeout must be (connect, read)"
    assert seen["timeout"][0] == transport.CONNECT_TIMEOUT
    assert seen["timeout"][0] < seen["timeout"][1]


# --------------------------------------------------------------------------
# 4. and the step has to stay soft whatever happens below it
# --------------------------------------------------------------------------

def test_the_agent_reports_a_failed_image_instead_of_raising(tmp_path, monkeypatch):
    from src.agents.image import generator
    from src.data.schemas import ImageBrief
    from src.models.image_gen import ImageService

    class Exploding:
        name, is_mock = "boom", False

        def is_available(self):
            return True

        def generate(self, request, output_path):
            raise ReadTimeout("Read timed out. (read timeout=120.0)")

    service = ImageService(provider=Exploding(), cache_dir=None)
    brief = ImageBrief(kind="amenity_concept", subject="a courtyard at dusk",
                       purpose="Shows the kind of shared garden the plans describe.",
                       caption="A quiet courtyard.", aspect="wide")
    report = generator.run([brief], generator.StyleProfile(), service, tmp_path)

    assert report.visuals == []
    assert report.skipped and "unexpectedly" in report.skipped[0].lower()


def test_a_quota_error_still_fails_over_rather_than_being_swallowed(tmp_path):
    """The backstop must not blunt the signals the chain depends on."""
    def spent(*_a, **_k):
        return _Response(status_code=429, headers={"content-type": "application/json"},
                         text='{"errors":[{"code":3036}]}')

    class _Json(_Response):
        def json(self):
            return {"errors": [{"code": 3036}]}

    provider = _cloudflare(lambda *_a, **_k: _Json(
        status_code=429, headers={"content-type": "application/json"}))
    with pytest.raises(ImageGenerationQuotaError):
        provider.generate(_request(), tmp_path / "out.png")
