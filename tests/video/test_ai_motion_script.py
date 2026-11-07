"""scripts/test_ai_motion.py: nothing is sent without --spend, the estimate is
itemised, the cap stops a call before it is sent, and a cached answer is free."""
from __future__ import annotations

import json

import pytest

from scripts import test_ai_motion as script
from src.core.config import settings
from src.models.video_gen import ImageToVideoRequest, selector
from src.models.video_gen.fal import FalClient, FalVideoProvider
from tests.video.fakes import make_photo

ARGS = ["--property-id", "test-prop", "--hook", "hook.jpg", "--court", "court.jpg", "--living", "living.jpg"]


@pytest.fixture
def prop():
    images = settings.data_dir_resolved / "properties" / "test-prop" / "images"
    for i, name in enumerate(("hook.jpg", "court.jpg", "living.jpg")):
        make_photo(images / name, (1920, 1015), seed=i)
    return images


@pytest.fixture
def no_submit(monkeypatch):
    sent = []

    def refuse(self, endpoint, payload):
        sent.append(endpoint)
        raise AssertionError("a paid call was sent")
    monkeypatch.setattr(FalClient, "submit", refuse)
    return sent


def _run(argv, **kwargs) -> tuple[int, str]:
    lines: list[str] = []
    code = script.main(argv, write=lines.append, **kwargs)
    return code, "\n".join(lines)


def test_without_spend_only_the_itemised_estimate_is_printed(prop, no_submit):
    code, out = _run(ARGS)
    assert code == 0 and no_submit == []
    assert "Nothing was sent. Run again with --spend" in out
    rows = [line for line in out.splitlines() if line[:3].strip().isdigit()]
    assert len(rows) == 13
    assert sum("fal-ai/flux-2-pro/outpaint" in r for r in rows) == 3
    assert sum("fal-ai/kling-video/v2.5-turbo/pro" in r for r in rows) == 1
    assert sum("minimax/h3-max/image-to-video" in r for r in rows) == 9
    assert sum("transition" in r for r in rows) == 2 and sum("travel" in r for r in rows) == 1
    assert "To pay: $2.989 for 13 call(s); 0 cached (free). Cap: $4.00." in out
    assert not (settings.data_dir_resolved / "video_cache" / "fal_motion").exists()


def test_spend_without_a_key_sends_nothing(prop, no_submit, monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    code, out = _run([*ARGS, "--spend"])
    assert code == 2 and "FAL_KEY is not set" in out and no_submit == []


def test_an_estimate_over_the_cap_is_refused_even_with_spend_and_a_key(prop, no_submit, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    code, out = _run([*ARGS, "--spend", "--cap", "1.0"])
    assert code == 2 and "over the cap of $1.00. Nothing was sent." in out and no_submit == []


def test_the_cap_cannot_be_raised_above_four_dollars(prop, no_submit):
    code, out = _run([*ARGS, "--cap", "100"])
    assert code == 0 and "Cap: $4.00." in out


def test_the_key_is_never_printed(prop, no_submit, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "secret-fal-key-value")
    code, out = _run([*ARGS, "--cap", "1.0"])
    assert "secret-fal-key-value" not in out


# ---------------------------------------------------------------------------
# The ledger

def _h3(api_key="k") -> FalVideoProvider:
    return FalVideoProvider(selector.load_catalog()["minimax_h3_max"], api_key, FalClient(api_key))


def _request(tmp_path) -> ImageToVideoRequest:
    image = make_photo(tmp_path / "kf.jpg", (72, 128), seed=4)
    return ImageToVideoRequest(image_path=image, prompt="push in", duration_s=3.0, width=72, height=128)


def test_a_call_that_would_pass_the_cap_is_stopped_before_it_is_sent(tmp_path, no_submit):
    ledger = script.Ledger(tmp_path / "cache", cap=0.5, spend=True)
    ledger.spent = 0.4
    with pytest.raises(script.CapReached, match="over the cap of \\$0.50"):
        ledger.video("shot", _h3(), _request(tmp_path), tmp_path / "out.mp4")
    assert no_submit == [] and ledger.calls[-1].state == "stopped"


def test_the_dry_run_prices_a_call_and_never_sends_it(tmp_path, no_submit):
    ledger = script.Ledger(tmp_path / "cache", cap=4.0, spend=False)
    with pytest.raises(script.NotMade):
        ledger.video("shot", _h3(), _request(tmp_path), tmp_path / "out.mp4")
    call = ledger.calls[-1]
    assert call.state == "to pay" and call.price == pytest.approx(0.296) and no_submit == []
    ledger.pending("later", "minimax/h3-max/image-to-video", "2 s", 0.216, "needs the shots first")
    assert ledger.to_pay == pytest.approx(0.512)


def test_a_cached_answer_is_free_and_not_sent(tmp_path, no_submit):
    provider, request = _h3(), _request(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    key = script.cache_key(provider.endpoint, provider.payload(request))
    (cache / f"{key}.mp4").write_bytes(b"clip")
    (cache / f"{key}.json").write_text(json.dumps({
        "provider": "minimax_h3_max", "model_id": provider.endpoint, "duration_s": 3.7, "width": 72, "height": 128,
        "fps": 24, "cost_usd": 0.296, "billed_s": 3.7, "request_id": "r1"}), encoding="utf-8")
    ledger = script.Ledger(cache, cap=0.1, spend=True)
    clip = ledger.video("shot", provider, request, tmp_path / "out.mp4")
    assert clip.from_cache and (tmp_path / "out.mp4").read_bytes() == b"clip"
    assert ledger.calls[-1].state == "cached" and ledger.calls[-1].price == 0.0 and ledger.spent == 0.0
    assert ledger.calls[-1].made_cost == pytest.approx(0.296) and no_submit == []


def test_the_cache_key_is_what_was_sent_with_pictures_as_digests(tmp_path):
    provider, request = _h3(), _request(tmp_path)
    a = script.cache_key(provider.endpoint, provider.payload(request))
    assert a == script.cache_key(provider.endpoint, provider.payload(request))
    request.prompt = "pull out"
    assert script.cache_key(provider.endpoint, provider.payload(request)) != a
    request.prompt = "push in"
    make_photo(request.image_path, (72, 128), seed=5)
    assert script.cache_key(provider.endpoint, provider.payload(request)) != a
