"""The chain and the cache: what a failed provider costs, and what a repeat costs."""
from __future__ import annotations

import pytest

from src.models.video_gen import (
    VideoContentRefusedError, VideoGenerationAuthError, VideoGenerationError, VideoGenerationQuotaError,
    VideoGenerationTransportError, VideoService, get_video_service, selector,
)
from src.models.video_gen.kenburns import KenBurnsProvider
from src.rendering import ffmpeg
from tests.video.fakes import ScriptedVideoProvider, make_photo, paid_spec, small_request


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


def _service(cache_dir, *providers, **kwargs):
    return VideoService(list(providers) + [KenBurnsProvider()], cache_dir, **kwargs)


class TestFallback:
    def test_a_quota_error_drops_the_provider_for_the_rest_of_the_run(self, photo, tmp_path, cache_dir):
        paid = ScriptedVideoProvider(script=[VideoGenerationQuotaError("balance is zero")])
        service = _service(cache_dir, paid)

        first = service.generate(small_request(photo), tmp_path / "shot0.mp4")
        second = service.generate(small_request(photo, camera="pan_left"), tmp_path / "shot1.mp4")

        assert first.provider == second.provider == "kenburns"
        assert "paid: allowance spent: balance is zero" in first.reason
        assert "dropped for this run" in second.reason
        assert len(paid.calls) == 1                       # never asked again
        assert set(service.dropped) == {"paid"}
        assert selector.quota_until("paid") is not None   # and the next run knows

    def test_an_auth_error_drops_the_provider_but_leaves_no_marker(self, photo, tmp_path, cache_dir):
        paid = ScriptedVideoProvider(script=[VideoGenerationAuthError("401")])
        service = _service(cache_dir, paid)

        clip = service.generate(small_request(photo), tmp_path / "a.mp4")
        service.generate(small_request(photo), tmp_path / "b.mp4")

        assert clip.provider == "kenburns" and "key rejected" in clip.reason
        assert len(paid.calls) == 1
        assert selector.quota_until("paid") is None

    def test_a_transport_error_costs_one_shot_and_the_provider_is_tried_again(self, photo, tmp_path, cache_dir):
        paid = ScriptedVideoProvider(script=[VideoGenerationTransportError("timed out")])
        service = _service(cache_dir, paid)

        first = service.generate(small_request(photo), tmp_path / "shot0.mp4")
        second = service.generate(small_request(photo, camera="pull_out"), tmp_path / "shot1.mp4")

        assert first.provider == "kenburns" and first.reason == "paid: timed out"
        assert second.provider == "paid" and second.reason == ""
        assert len(paid.calls) == 2 and service.dropped == {}

    def test_a_refusal_falls_back_for_that_shot_without_dropping_the_provider(self, photo, tmp_path, cache_dir):
        paid = ScriptedVideoProvider(script=[VideoContentRefusedError("declined")])
        service = _service(cache_dir, paid)

        clip = service.generate(small_request(photo), tmp_path / "a.mp4")

        assert clip.provider == "kenburns" and "declined" in clip.reason
        assert service.dropped == {}

    def test_a_provider_with_no_key_is_skipped_and_says_so(self, photo, tmp_path, cache_dir):
        keyless = ScriptedVideoProvider(spec=paid_spec(key_env="PAID_KEY"))
        clip = _service(cache_dir, keyless).generate(small_request(photo), tmp_path / "a.mp4")
        assert clip.provider == "kenburns" and clip.reason == "paid: not available"
        assert keyless.calls == []

    def test_the_fallback_clip_is_a_real_video_of_the_asked_size(self, photo, tmp_path, cache_dir):
        paid = ScriptedVideoProvider(script=[VideoGenerationTransportError("down")])
        clip = _service(cache_dir, paid).generate(small_request(photo), tmp_path / "a.mp4")
        info = ffmpeg.probe(tmp_path / "a.mp4")
        assert (info.width, info.height, info.vcodec) == (160, 284, "h264")
        assert clip.path == tmp_path / "a.mp4"

    def test_the_note_about_an_unavailable_choice_travels_with_the_clip(self, photo, tmp_path, cache_dir):
        service = VideoService([KenBurnsProvider()], cache_dir, note="Seedance 2.0 is unavailable (no key)")
        clip = service.generate(small_request(photo), tmp_path / "a.mp4")
        assert clip.reason == "Seedance 2.0 is unavailable (no key)"

    def test_when_nothing_can_make_the_clip_the_error_says_what_each_said(self, tmp_path, cache_dir):
        not_a_picture = tmp_path / "bad.jpg"
        not_a_picture.write_text("not an image", encoding="utf-8")
        paid = ScriptedVideoProvider(script=[VideoGenerationTransportError("down")])
        with pytest.raises(VideoGenerationError, match=r"paid: down.*kenburns: cannot read bad.jpg"):
            _service(cache_dir, paid).generate(small_request(not_a_picture), tmp_path / "a.mp4")

    def test_a_missing_picture_is_reported_once_without_asking_any_provider(self, tmp_path, cache_dir):
        paid = ScriptedVideoProvider()
        with pytest.raises(VideoGenerationError, match="the input picture does not exist"):
            _service(cache_dir, paid).generate(small_request(tmp_path / "missing.jpg"), tmp_path / "a.mp4")
        assert paid.calls == []


class TestCache:
    def test_the_same_bytes_under_another_path_are_a_hit(self, photo, tmp_path, cache_dir):
        paid = ScriptedVideoProvider()
        service = _service(cache_dir, paid)
        first = service.generate(small_request(photo), tmp_path / "out" / "one.mp4")

        copy = tmp_path / "elsewhere" / "renamed.jpg"
        copy.parent.mkdir()
        copy.write_bytes(photo.read_bytes())
        again = service.generate(small_request(copy), tmp_path / "out" / "two.mp4")

        assert (first.from_cache, again.from_cache) == (False, True)
        assert len(paid.calls) == 1
        assert again.cost_usd == 0.0 and first.cost_usd == pytest.approx(0.1)
        assert (tmp_path / "out" / "two.mp4").exists() and again.path == tmp_path / "out" / "two.mp4"

    def test_changed_bytes_are_a_miss(self, photo, tmp_path, cache_dir):
        paid = ScriptedVideoProvider()
        service = _service(cache_dir, paid)
        service.generate(small_request(photo), tmp_path / "one.mp4")

        make_photo(photo, (1600, 900), seed=99)            # same path, different picture
        clip = service.generate(small_request(photo), tmp_path / "two.mp4")

        assert clip.from_cache is False and len(paid.calls) == 2

    @pytest.mark.parametrize("change", [
        {"camera": "pan_left"}, {"prompt": "slow drift"}, {"duration_s": 2.0}, {"width": 180},
        {"fps": 12}, {"seed": 7}, {"focal": "20% 30%"},
    ])
    def test_any_change_to_the_request_is_a_miss(self, photo, tmp_path, cache_dir, change):
        paid = ScriptedVideoProvider()
        service = _service(cache_dir, paid)
        service.generate(small_request(photo), tmp_path / "one.mp4")
        service.generate(small_request(photo, **change), tmp_path / "two.mp4")
        assert len(paid.calls) == 2

    def test_a_different_provider_does_not_share_entries(self, photo, tmp_path, cache_dir):
        one, two = ScriptedVideoProvider("one"), ScriptedVideoProvider("two")
        _service(cache_dir, one).generate(small_request(photo), tmp_path / "a.mp4")
        _service(cache_dir, two).generate(small_request(photo), tmp_path / "b.mp4")
        assert len(one.calls) == len(two.calls) == 1

    def test_a_cache_hit_needs_no_key(self, photo, tmp_path, cache_dir):
        _service(cache_dir, ScriptedVideoProvider()).generate(small_request(photo), tmp_path / "a.mp4")
        keyless = ScriptedVideoProvider(spec=paid_spec(key_env="PAID_KEY"))
        clip = _service(cache_dir, keyless).generate(small_request(photo), tmp_path / "b.mp4")
        assert clip.from_cache and keyless.calls == []

    def test_pan_and_zoom_is_not_cached(self, photo, tmp_path, cache_dir):
        clip = VideoService([KenBurnsProvider()], cache_dir).generate(small_request(photo), tmp_path / "a.mp4")
        assert clip.from_cache is False
        assert not cache_dir.exists() or list(cache_dir.glob("*.mp4")) == []

    def test_a_half_written_entry_is_ignored(self, photo, tmp_path, cache_dir):
        paid = ScriptedVideoProvider()
        service = _service(cache_dir, paid)
        key = service._key(paid, small_request(photo))
        cache_dir.mkdir()
        (cache_dir / f"{key}.mp4").write_bytes(b"partial")      # no .json beside it
        clip = service.generate(small_request(photo), tmp_path / "a.mp4")
        assert clip.from_cache is False and len(paid.calls) == 1


class TestBuilding:
    def test_pan_and_zoom_builds_one_provider_and_a_cache_folder_resolved_now(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.core.config.settings.data_dir", str(tmp_path / "elsewhere"), raising=False)
        service = get_video_service("kenburns", 15)
        assert [p.id for p in service.providers] == ["kenburns"]
        assert service.cache_dir == (tmp_path / "elsewhere").resolve() / "video_cache"

    def test_auto_with_no_key_ends_up_on_pan_and_zoom(self):
        service = get_video_service("auto", 15)
        assert [p.id for p in service.providers] == ["kenburns"]
        assert service.estimated_cost_usd == 0.0

    def test_a_named_model_that_cannot_run_carries_its_reason(self):
        service = get_video_service("seedance_2", 15)
        assert [p.id for p in service.providers] == ["kenburns"]
        assert "adapter not installed" in service.note
