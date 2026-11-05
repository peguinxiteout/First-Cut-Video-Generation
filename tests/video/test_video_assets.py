"""Node 3: which pictures can carry a shot, and how they rank."""
from __future__ import annotations

import pytest
from PIL import Image, ImageFilter

from src.agents.video import assets as asset_audit
from src.agents.video import brief as brief_agent
from tests.video.fakes import make_photo
from tests.video.helpers import make_video_property


@pytest.fixture
def refuse_vision(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the vision model was asked")

    monkeypatch.setattr("src.agents.brochure.vision.call_with_fallback", refuse)


def _images(folder):
    return brief_agent.run(folder).images


def test_sharpness_tells_a_sharp_picture_from_a_blurred_one(tmp_path):
    sharp = make_photo(tmp_path / "sharp.jpg", (800, 600), seed=3)
    Image.open(sharp).filter(ImageFilter.GaussianBlur(8)).save(tmp_path / "blurred.jpg")

    _, _, sharp_variance = asset_audit.measure(str(sharp))
    width, height, blurred_variance = asset_audit.measure(str(tmp_path / "blurred.jpg"))

    assert (width, height) == (800, 600)
    assert sharp_variance > 10 * blurred_variance
    assert asset_audit.sharpness_score(sharp_variance) > asset_audit.sharpness_score(blurred_variance)


def test_sharpness_scores_run_from_flat_to_ten():
    assert asset_audit.sharpness_score(0) == 0
    assert asset_audit.sharpness_score(1000) == pytest.approx(10.0, abs=0.1)
    assert 6 < asset_audit.sharpness_score(100) < 7.5
    assert asset_audit.sharpness_score(10**9) == 10


def test_an_unreadable_file_measures_as_nothing(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image", encoding="utf-8")
    assert asset_audit.measure(str(bad)) == (0, 0, 0.0)


def test_photographs_are_usable_and_plans_and_logos_are_not(tmp_path, refuse_vision):
    analyses, ranked, plan, notes = asset_audit.run(_images(make_video_property(tmp_path)), offline=True, photo_shots=4)

    usable = {a.filename for a in ranked if a.usable}
    assert "exterior_facade.jpg" in usable and "swimming_pool.jpg" in usable
    by_name = {a.filename: a for a in ranked}
    assert not by_name["floorplan_plan_0.png"].usable and "not a photograph" in by_name["floorplan_plan_0.png"].note
    assert not by_name["logo_developer_logo.png"].usable
    assert all(a.usable for a in ranked[:len(usable)])             # usable ones first
    assert notes[0] == f"{len(analyses)} pictures read from the uploads (offline run)."
    assert "8 usable photographs" in notes[1]


def test_what_a_picture_shows_comes_from_its_name_when_no_model_looks(tmp_path, refuse_vision):
    _, ranked, _, _ = asset_audit.run(_images(make_video_property(tmp_path)), offline=True, photo_shots=4)
    topics = {a.filename: a.topic for a in ranked}
    assert topics["swimming_pool.jpg"] == "pool" and topics["exterior_facade.jpg"] == "exterior"
    assert topics["clubhouse.jpg"] == "clubhouse"


def test_a_label_the_person_gave_a_photograph_decides_what_it_shows(tmp_path, refuse_vision):
    import json

    folder = make_video_property(tmp_path, photos=2)
    make_photo(folder / "images" / "view_a.jpg", (1200, 800), seed=50)
    roles_path = folder / "images" / "roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    roles["view_a.jpg"] = "photo:pool"
    roles_path.write_text(json.dumps(roles), encoding="utf-8")

    _, ranked, _, _ = asset_audit.run(_images(folder), offline=True, photo_shots=3)

    assert next(a for a in ranked if a.filename == "view_a.jpg").topic == "pool"


def test_every_asset_carries_its_measurements(tmp_path, refuse_vision):
    _, ranked, _, _ = asset_audit.run(_images(make_video_property(tmp_path)), offline=True, photo_shots=4)
    photo = next(a for a in ranked if a.filename == "exterior_facade.jpg")
    assert (photo.width, photo.height) == (1600, 1000)
    assert photo.sharpness > 0 and 0 < photo.quality <= 10 and photo.aspect == pytest.approx(1.6)


def test_a_picture_too_small_to_fill_a_frame_is_set_aside_with_the_size(tmp_path, refuse_vision):
    folder = make_video_property(tmp_path, photos=3)
    make_photo(folder / "images" / "tiny_pool.jpg", (400, 300), seed=40)
    _, ranked, _, notes = asset_audit.run(_images(folder), offline=True, photo_shots=3)
    tiny = next(a for a in ranked if a.filename == "tiny_pool.jpg")
    assert not tiny.usable and tiny.note == "too small (400x300) to fill a frame"
    assert "tiny_pool.jpg (too small (400x300) to fill a frame)" in notes[1]


def test_a_twin_picture_is_marked_as_a_duplicate_of_the_better_one(tmp_path, refuse_vision):
    folder = make_video_property(tmp_path, photos=3)
    original = folder / "images" / "exterior_facade.jpg"
    (folder / "images" / "exterior_facade_copy.jpg").write_bytes(original.read_bytes())

    _, ranked, plan, _ = asset_audit.run(_images(folder), offline=True, photo_shots=3)

    twins = [a for a in ranked if a.filename.startswith("exterior_facade")]
    assert len(twins) == 2 and twins[0].cluster == twins[1].cluster >= 0
    kept, dropped = [a for a in twins if a.usable], [a for a in twins if not a.usable]
    assert len(kept) == 1 and len(dropped) == 1
    assert dropped[0].duplicate_of == kept[0].image_id and dropped[0].note.startswith("looks like ")


def test_fewer_usable_pictures_than_shots_is_flagged(tmp_path, refuse_vision):
    _, _, _, notes = asset_audit.run(_images(make_video_property(tmp_path, photos=2)), offline=True, photo_shots=5)
    flag = [n for n in notes if n.startswith("Only 2 usable photograph(s) for 5 photo shots")]
    assert flag and "different crop" in flag[0]


def test_enough_pictures_is_not_flagged(tmp_path, refuse_vision):
    _, _, _, notes = asset_audit.run(_images(make_video_property(tmp_path)), offline=True, photo_shots=4)
    assert not any(n.startswith("Only ") for n in notes)


def test_offline_the_vision_model_is_never_asked(tmp_path, refuse_vision, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "a-key-that-must-not-be-used", raising=False)
    asset_audit.run(_images(make_video_property(tmp_path)), offline=True, photo_shots=4)      # would raise if asked


def test_online_a_failing_vision_model_falls_back_to_the_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "k", raising=False)

    def fail(images):
        raise RuntimeError("vision is down")

    monkeypatch.setattr("src.agents.video.vision_cache.classify_batch", fail)

    _, ranked, _, notes = asset_audit.run(_images(make_video_property(tmp_path)), offline=False, photo_shots=4)

    assert "vision unavailable: RuntimeError" in notes[0]
    assert any(a.usable for a in ranked)


def test_the_blended_quality_is_written_back_for_the_planner(tmp_path, refuse_vision):
    images = _images(make_video_property(tmp_path, photos=2))
    analyses, ranked, _, _ = asset_audit.run(images, offline=True, photo_shots=2)
    for asset in ranked:
        assert analyses[asset.image_id].quality == asset.quality


def test_a_pictures_caption_adds_the_tags_its_words_name_without_a_second_model_call(tmp_path, monkeypatch):
    from src.agents.video import vision_cache
    from src.data.video_schemas import PhotoAnalysis

    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "k", raising=False)
    caption = "Outdoor fitness equipment and children play structures set in a landscaped garden"

    def reading(images, **kwargs):
        return vision_cache.Reading(analyses={i.image_id: PhotoAnalysis(
            image_id=i.image_id, category="amenity", quality=7, composition=7, hero_score=6, caption=caption,
            tags=["amenity", "play area", "lawn"]) for i in images})

    monkeypatch.setattr("src.agents.video.vision_cache.classify", reading)
    _, ranked, _, _ = asset_audit.run(_images(make_video_property(tmp_path)), offline=False, photo_shots=4)
    assert ranked and all(a.tags == ["amenity", "play area", "lawn", "gym"] and a.caption == caption for a in ranked)
