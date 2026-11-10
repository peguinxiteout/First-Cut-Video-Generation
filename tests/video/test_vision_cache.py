"""Vision for the asset audit: remembered per picture, asked in bounded parallel batches."""
from __future__ import annotations

import threading
import time

import pytest

from src.agents.video import assets as asset_audit
from src.agents.video import policy, vision_cache
from src.core.config import load_yaml_config
from src.data.schemas import ImageAnalysis, ImageAsset
from src.models import gateway
from tests.video.fakes import make_photo


def _analysis(image_id, caption="a view") -> ImageAnalysis:
    return ImageAnalysis(image_id=image_id, category="exterior", quality=7.5, composition=7.0, hero_score=6.0,
                         suggested_caption=caption)


def _images(tmp_path, n=7, same=()) -> list[ImageAsset]:
    out = []
    for i in range(n):
        seed = 0 if i in same else i + 1                     # pictures in `same` are byte-identical twins
        path = make_photo(tmp_path / "photos" / f"p{i}.jpg", (400, 300), seed=seed)
        out.append(ImageAsset(image_id=f"IMG{i:03d}", filename=path.name, path=str(path), role="photo"))
    return out


class Runner:
    """A vision batch that answers every picture it is given, and remembers what it was asked."""

    def __init__(self, delay=0.0, fail_on=(), caption="read"):
        self.batches: list[list[str]] = []
        self.delay, self.fail_on, self.caption = delay, set(fail_on), caption
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def __call__(self, chunk):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.batches.append([i.image_id for i in chunk])
        try:
            time.sleep(self.delay)
            if self.fail_on & {i.image_id for i in chunk}:
                raise RuntimeError("batch refused")
            return {i.image_id: _analysis(i.image_id, self.caption) for i in chunk}
        finally:
            with self.lock:
                self.active -= 1


def _classify(images, runner, tmp_path, **kw):
    kw.setdefault("concurrency", 3)
    kw.setdefault("batch_size", 3)
    return vision_cache.classify(images, runner=runner, directory=tmp_path / "cache", model="gemini:m",
                                 version="v1", **kw)


# -- the cache ---------------------------------------------------------------------------------------------------

def test_a_second_run_of_the_same_pictures_asks_the_model_nothing(tmp_path):
    images, runner = _images(tmp_path), Runner()
    first = _classify(images, runner, tmp_path)
    again = _classify(images, runner, tmp_path)

    assert (first.cached, first.asked) == (0, 7) and (again.cached, again.asked) == (7, 0)
    assert len(runner.batches) == 3                                     # 7 pictures in batches of 3: only the first run asked
    assert {k: v.model_dump() for k, v in again.analyses.items()} == {k: v.model_dump() for k, v in first.analyses.items()}
    assert again.notes == ["Vision: 7 from the cache."]


def test_only_a_new_picture_is_asked_about(tmp_path):
    images, runner = _images(tmp_path), Runner()
    _classify(images[:5], runner, tmp_path)
    runner.batches.clear()
    reading = _classify(images, runner, tmp_path)
    assert reading.cached == 5 and sorted(i for b in runner.batches for i in b) == ["IMG005", "IMG006"]


def test_the_reading_is_kept_under_the_pictures_bytes_not_its_id_or_name(tmp_path):
    images, runner = _images(tmp_path, 2), Runner()
    _classify(images, runner, tmp_path)
    renamed = [ImageAsset(image_id=f"NEW{i}", filename=f"other{i}.jpg", path=im.path, role="photo")
               for i, im in enumerate(images)]
    reading = _classify(renamed, runner, tmp_path)
    assert reading.cached == 2 and set(reading.analyses) == {"NEW0", "NEW1"}
    assert all(a.image_id == k for k, a in reading.analyses.items())


def test_a_changed_picture_a_new_model_or_a_new_prompt_is_read_afresh(tmp_path):
    images, runner = _images(tmp_path, 3), Runner()
    _classify(images, runner, tmp_path)
    for kw in ({"model": "gemini:other"}, {"version": "v2"}):
        args = dict(runner=runner, directory=tmp_path / "cache", model="gemini:m", version="v1", concurrency=3, batch_size=3)
        args.update(kw)
        assert vision_cache.classify(images, **args).cached == 0
    make_photo(tmp_path / "photos" / "p0.jpg", (400, 300), seed=99)                       # the picture's bytes change
    assert _classify(images, runner, tmp_path).cached == 2


def test_the_key_is_the_bytes_the_model_and_the_prompt_version():
    a = vision_cache.cache_key(b"picture", "gemini:m", "v1")
    assert a == vision_cache.cache_key(b"picture", "gemini:m", "v1")
    assert len({a, vision_cache.cache_key(b"other", "gemini:m", "v1"), vision_cache.cache_key(b"picture", "gemini:n", "v1"),
                vision_cache.cache_key(b"picture", "gemini:m", "v2")}) == 4


def test_the_prompt_version_follows_the_prompts_text_and_the_model_is_the_roles(monkeypatch):
    real = vision_cache.load_yaml_config
    version = vision_cache.prompt_version()
    monkeypatch.setattr(vision_cache, "load_yaml_config", lambda name: {**real(name), "vision_classify_batch": "another prompt"}
                        if name == "brochure/prompts.yaml" else real(name))
    assert vision_cache.prompt_version() != version
    monkeypatch.setattr("src.core.config.settings.gemini_vision_model", "gemini-x", raising=False)
    assert vision_cache.model_name() == "gemini:gemini-x"


def test_identical_pictures_are_asked_about_once_and_both_get_the_reading(tmp_path):
    images, runner = _images(tmp_path, 4, same=(0, 2)), Runner()
    reading = _classify(images, runner, tmp_path)
    assert sum(len(b) for b in runner.batches) == 3 and reading.asked == 3
    assert set(reading.analyses) == {i.image_id for i in images}
    assert reading.analyses["IMG002"].image_id == "IMG002"


def test_an_unreadable_cache_entry_is_asked_again(tmp_path):
    images, runner = _images(tmp_path, 2), Runner()
    _classify(images, runner, tmp_path)
    for entry in (tmp_path / "cache").glob("*.json"):
        entry.write_text("{not json", encoding="utf-8")
    assert _classify(images, runner, tmp_path).asked == 2


def test_the_cache_lives_in_the_data_folder_and_is_not_in_git():
    assert vision_cache.cache_dir().name == "vision_cache"
    assert "data/vision_cache/" in (vision_cache.Path(__file__).resolve().parents[2] / ".gitignore").read_text(encoding="utf-8")


# -- the parallel batches ----------------------------------------------------------------------------------------------

def test_pictures_are_split_into_batches_of_the_configured_size(tmp_path):
    runner = Runner()
    _classify(_images(tmp_path, 7), runner, tmp_path, batch_size=3)
    assert sorted(len(b) for b in runner.batches) == [1, 3, 3]
    assert sorted(i for b in runner.batches for i in b) == [f"IMG{i:03d}" for i in range(7)]


def test_no_more_batches_than_the_limit_are_in_flight_at_once(tmp_path):
    runner = Runner(delay=0.15)
    _classify(_images(tmp_path, 10), runner, tmp_path, concurrency=3, batch_size=1)
    assert runner.peak == 3 and len(runner.batches) == 10


def test_the_limit_is_configurable_and_one_means_one_at_a_time(tmp_path):
    runner = Runner(delay=0.05)
    _classify(_images(tmp_path, 4), runner, tmp_path, concurrency=1, batch_size=1)
    assert runner.peak == 1


def test_batches_run_in_parallel_not_one_after_another(tmp_path):
    runner = Runner(delay=0.3)
    started = time.perf_counter()
    _classify(_images(tmp_path, 6), runner, tmp_path, concurrency=3, batch_size=2)
    assert time.perf_counter() - started < 0.75                         # three 0.3 s batches took about 0.3 s, not 0.9


def test_the_shipped_settings_are_three_at_a_time():
    config = load_yaml_config("video/vision.yaml")
    assert config["concurrency"] == 3 and config["batch_size"] >= 1


def test_the_gateways_call_policy_travels_into_every_batch(tmp_path):
    seen = []

    def runner(chunk):
        seen.append(gateway._call_policy.get())
        return {i.image_id: _analysis(i.image_id) for i in chunk}

    with gateway.restricted(vendors=["gemini"]) as call_policy:
        _classify(_images(tmp_path, 6), runner, tmp_path, batch_size=2)
    assert len(seen) == 3 and all(p is call_policy for p in seen)


# -- a batch that fails --------------------------------------------------------------------------------------------------

def test_a_failing_batch_costs_only_its_own_pictures_and_the_rest_are_kept_and_cached(tmp_path):
    images = _images(tmp_path, 6)
    runner = Runner(fail_on={"IMG001"})
    reading = _classify(images, runner, tmp_path, batch_size=2)

    assert set(reading.failed) == {"IMG000", "IMG001"} and "RuntimeError: batch refused" in reading.failed["IMG001"]
    assert set(reading.analyses) == {"IMG002", "IMG003", "IMG004", "IMG005"}
    assert any("gave no reading for 2 picture(s)" in n for n in reading.notes)
    again = _classify(images, Runner(), tmp_path, batch_size=2)
    assert again.cached == 4 and again.asked == 2                      # the failed pair is asked again, the rest remembered


def test_a_picture_the_model_left_out_is_reported_not_invented(tmp_path):
    images = _images(tmp_path, 3)

    def runner(chunk):
        return {i.image_id: _analysis(i.image_id) for i in chunk if i.image_id != "IMG001"}

    reading = _classify(images, runner, tmp_path, batch_size=3)
    assert reading.failed == {"IMG001": "the model gave no reading for it"} and "IMG001" not in reading.analyses


def test_when_every_batch_fails_the_first_error_is_raised(tmp_path):
    with pytest.raises(RuntimeError, match="batch refused"):
        _classify(_images(tmp_path, 4), Runner(fail_on={f"IMG{i:03d}" for i in range(4)}), tmp_path, batch_size=2)


def test_a_picture_that_cannot_be_read_is_reported(tmp_path):
    images = _images(tmp_path, 2)
    images.append(ImageAsset(image_id="GONE", filename="gone.jpg", path=str(tmp_path / "gone.jpg"), role="photo"))
    reading = _classify(images, Runner(), tmp_path)
    assert reading.failed == {"GONE": "the picture could not be read"} and len(reading.analyses) == 2


# -- in the asset audit ----------------------------------------------------------------------------------------------------

def test_the_asset_audit_uses_the_cache_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "k", raising=False)
    calls = []

    def batch(chunk):
        calls.append(len(chunk))
        return {i.image_id: _analysis(i.image_id) for i in chunk}

    monkeypatch.setattr("src.agents.video.vision_cache.classify_batch", batch)
    images = _images(tmp_path, 5)
    _, _, _, first = asset_audit.run(images, offline=False, photo_shots=3)
    asked = sum(calls)
    _, _, _, second = asset_audit.run(images, offline=False, photo_shots=3)

    assert asked == 5 and sum(calls) == 5                                # nothing asked the second time
    assert "read by a vision model" in first[0] and "5 read by the model" in first[0]
    assert "read by a vision model (Vision: 5 from the cache.)" in second[0]
    assert policy.active() is None
