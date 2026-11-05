"""The light colour grade: photographs only, configurable, and off when asked."""
from __future__ import annotations

import pytest

from src.agents.video import brief as brief_agent
from src.rendering import ffmpeg, video_compose
from src.rendering.video_compose import ClipPart
from tests.video.helpers import make_video_property
from tests.video.test_transitions import pacing  # noqa: F401 - the fixture
from tests.video.test_video_compose import FPS, W, H, _compose, _frame

REAL_LOADER = video_compose.load_yaml_config


def _muted(path, seconds=2, colour="0x907050"):
    ffmpeg.run(["-f", "lavfi", "-i", f"color=c={colour}:s={W}x{H}:r={FPS}:d={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(path)])
    return path


def _spread(pixel) -> int:
    return max(pixel) - min(pixel)


def _render(tmp_path, name, **kw):
    folder = tmp_path / name
    folder.mkdir()
    clips = [ClipPart(_muted(folder / "photo.mp4"), 20, grade=True), ClipPart(_muted(folder / "card.mp4"), 20, grade=False)]
    result = _compose(folder, clips=clips, overlays=[], voices=[], **kw)
    return _frame(result.video_path, 1.0, folder, "photo").getpixel((135, 240)), _frame(result.video_path, 3.0, folder, "card").getpixel((135, 240))


# -- the filter ------------------------------------------------------------------------------------------------

def test_the_grade_is_an_ffmpeg_eq_from_the_config():
    config = {"enabled": True, "contrast": 1.05, "saturation": 1.1, "brightness": 0.0}
    assert video_compose.grade_filter(config) == "eq=contrast=1.05:saturation=1.1:brightness=0.0"
    assert video_compose.grade_filter({**config, "enabled": False}) == ""
    assert video_compose.grade_filter({}) == ""


def test_the_shipped_grade_is_light():
    grade = REAL_LOADER("video/pacing.yaml")["compose"]["grade"]
    assert grade["enabled"] is True
    assert 1.0 < grade["contrast"] <= 1.1 and 1.0 < grade["saturation"] <= 1.15 and abs(grade["brightness"]) <= 0.05


# -- in the picture --------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_a_photograph_is_graded_and_a_card_keeps_the_brands_exact_colours(tmp_path, pacing):  # noqa: F811
    pacing(transition_s=0)
    photo_on, card_on = _render(tmp_path, "on")
    photo_off, card_off = _render(tmp_path, "off", grade=False)

    assert _spread(photo_on) > _spread(photo_off)                       # more saturated
    assert photo_on != photo_off
    assert all(abs(a - b) <= 2 for a, b in zip(card_on, card_off))     # the card is not touched


@pytest.mark.slow
def test_the_grade_can_be_switched_off_for_every_video_in_the_config(tmp_path, pacing):  # noqa: F811
    pacing(transition_s=0, grade={"enabled": False, "contrast": 1.2, "saturation": 1.5, "brightness": 0.0})
    photo, _ = _render(tmp_path, "cfg_off")
    pacing(transition_s=0)
    plain, _ = _render(tmp_path, "plain", grade=False)
    assert all(abs(a - b) <= 2 for a, b in zip(photo, plain))


@pytest.mark.slow
def test_the_amounts_come_from_the_config(tmp_path, pacing):  # noqa: F811
    pacing(transition_s=0, grade={"enabled": True, "contrast": 1.0, "saturation": 1.0, "brightness": 0.0})
    neutral, _ = _render(tmp_path, "neutral")
    pacing(transition_s=0, grade={"enabled": True, "contrast": 1.0, "saturation": 1.8, "brightness": 0.0})
    vivid, _ = _render(tmp_path, "vivid")
    assert _spread(vivid) > _spread(neutral) + 20


def test_a_video_turns_its_own_grade_off_in_video_input_json(tmp_path):
    assert brief_agent.run(make_video_property(tmp_path / "a"), draft=True).brief.grade is True
    off = brief_agent.run(make_video_property(tmp_path / "b", video_input={"grade": False}), draft=True)
    assert off.brief.grade is False
