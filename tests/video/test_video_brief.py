from __future__ import annotations

import json

import pytest

from src.agents.video import brief as brief_agent
from src.core.exceptions import IngestionError
from tests.video.helpers import make_video_property


def test_a_property_with_no_video_input_json_gets_every_default(tmp_path):
    folder = make_video_property(tmp_path)
    assert not (folder / "video_input.json").exists()

    loaded = brief_agent.run(folder)

    b = loaded.brief
    assert (b.duration_s, b.aspect, b.objective, b.model, b.allow_generated) == (15, "9:16", "awareness", "auto", False)
    assert (b.width, b.height, b.fps, b.quality) == (1080, 1920, 25, "final")
    assert b.property_id == loaded.property_input.property_id and b.name == "Aurelia Heights"
    assert b.category == "mid" and b.lang == "en"
    assert b.cta_text == "Book a site visit"             # the property's own call to action
    assert b.tone == "warm" and b.music_mood == "warm acoustic"


def test_video_input_json_is_read_and_unknown_keys_are_ignored(tmp_path):
    folder = make_video_property(tmp_path, video_input={
        "duration_s": 10, "aspect": "16:9", "objective": "Site Visit", "highlights": ["  Rooftop deck ", ""],
        "tone": "Refined", "cta": "Call us", "note": "hand-edited"})

    b = brief_agent.run(folder).brief

    assert (b.duration_s, b.aspect, b.objective, b.tone, b.cta_text) == (10, "16:9", "site_visit", "refined", "Call us")
    assert b.highlights == ["Rooftop deck"]
    assert (b.width, b.height) == (1920, 1080)


def test_command_line_options_win_over_the_file_but_empty_ones_do_not(tmp_path):
    folder = make_video_property(tmp_path, video_input={"duration_s": 30, "aspect": "1:1"})

    b = brief_agent.run(folder, {"duration_s": 10, "aspect": None, "model": ""}).brief

    assert (b.duration_s, b.aspect, b.model) == (10, "1:1", "auto")


@pytest.mark.parametrize("aspect, final, draft", [
    ("9:16", (1080, 1920), (540, 960)), ("16:9", (1920, 1080), (960, 540)), ("1:1", (1080, 1080), (540, 540))])
def test_the_size_follows_the_aspect_and_the_quality(tmp_path, aspect, final, draft):
    folder = make_video_property(tmp_path)
    big = brief_agent.run(folder, {"aspect": aspect}).brief
    small = brief_agent.run(folder, {"aspect": aspect}, draft=True).brief
    assert (big.width, big.height, big.fps) == (*final, 25)
    assert (small.width, small.height, small.fps, small.quality) == (*draft, 12, "draft")


@pytest.mark.parametrize("seconds, words", [(10, 17), (15, 27), (30, 59), (45, 90), (60, 122)])
def test_the_narration_budget_is_about_two_words_a_second_of_the_narrated_time(tmp_path, seconds, words):
    folder = make_video_property(tmp_path)
    assert brief_agent.run(folder, {"duration_s": seconds}).brief.target_words == words


def test_the_budget_leaves_out_the_closing_card(tmp_path):
    # 15 s less the 2 s card is 13 narrated seconds, at 2.1 words a second.
    assert brief_agent.run(make_video_property(tmp_path)).brief.target_words == int(13 * 2.1 + 0.5)


def test_hero_image_is_read_trimmed_and_defaults_to_none(tmp_path):
    folder = make_video_property(tmp_path, video_input={"hero_image": "  swimming_pool.jpg "})
    assert brief_agent.run(folder).brief.hero_image == "swimming_pool.jpg"
    assert brief_agent.run(make_video_property(tmp_path / "x")).brief.hero_image == ""


def test_shot_count_follows_the_pace_and_includes_the_closing_card(tmp_path):
    mid = brief_agent.run(make_video_property(tmp_path / "a", "Mid-Tier"), {"duration_s": 15}).brief
    brisk = brief_agent.run(make_video_property(tmp_path / "b", "Affordable"), {"duration_s": 15}).brief
    calm = brief_agent.run(make_video_property(tmp_path / "c", "Luxury"), {"duration_s": 15}).brief
    assert (mid.n_shots, brisk.n_shots, calm.n_shots) == (5, 6, 5)       # 3.0, 2.5 and 3.6 s a photograph
    assert brief_agent.run(make_video_property(tmp_path / "d"), {"duration_s": 10}).brief.n_shots == 4


def test_a_tone_sets_the_pace_over_the_category(tmp_path):
    folder = make_video_property(tmp_path, "Mid-Tier")
    shots = {tone: brief_agent.run(folder, {"duration_s": 30, "tone": tone}).brief.n_shots
             for tone in ("warm", "confident", "refined")}
    assert shots == {"warm": 10, "confident": 12, "refined": 9}


def test_an_unknown_tone_falls_back_to_the_categorys_and_says_so(tmp_path):
    loaded = brief_agent.run(make_video_property(tmp_path, "Luxury"), {"tone": "playful"})
    assert loaded.brief.tone == "refined"
    assert any("playful" in n for n in loaded.notes)


def test_an_explicit_call_to_action_beats_the_propertys_and_the_default(tmp_path):
    folder = make_video_property(tmp_path)
    assert brief_agent.run(folder, {"cta": "Visit this weekend"}).brief.cta_text == "Visit this weekend"
    blank = make_video_property(tmp_path / "x", cta="")
    assert brief_agent.run(blank, {"objective": "enquiry"}).brief.cta_text == "Enquire now"


def test_the_brand_kit_carries_the_logo_file_the_palette_and_the_identifiers(tmp_path):
    loaded = brief_agent.run(make_video_property(tmp_path))
    kit = loaded.brand_kit
    assert kit.logo_path and kit.logo_path.endswith("developer_logo.png")
    assert kit.palette_name == loaded.brief.palette_name and kit.palette["paper"].startswith("#")
    assert kit.accent.startswith("#")
    assert kit.rera_number == "P52100012345" and kit.phone == "+91 98765 43210"
    assert kit.website


def test_no_logo_means_no_logo_path(tmp_path):
    assert brief_agent.run(make_video_property(tmp_path, logo=False)).brand_kit.logo_path is None


@pytest.mark.parametrize("overrides", [{"duration_s": 20}, {"aspect": "4:3"}, {"objective": "viral"}])
def test_a_bad_option_is_an_ingestion_error_in_words(tmp_path, overrides):
    with pytest.raises(IngestionError, match="video options are not valid"):
        brief_agent.run(make_video_property(tmp_path), overrides)


def test_a_broken_video_input_json_is_named_in_the_error(tmp_path):
    folder = make_video_property(tmp_path)
    (folder / "video_input.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(IngestionError, match="video_input.json.*not valid JSON"):
        brief_agent.run(folder)
    (folder / "video_input.json").write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(IngestionError, match="must be a JSON object"):
        brief_agent.run(folder)


def test_a_property_without_pictures_is_an_ingestion_error(tmp_path):
    folder = make_video_property(tmp_path)
    for picture in (folder / "images").iterdir():
        picture.unlink()
    with pytest.raises(IngestionError):
        brief_agent.run(folder)
