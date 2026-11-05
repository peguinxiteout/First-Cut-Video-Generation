"""Crossfades between shots: where they go, and that they never change the length of the video."""
from __future__ import annotations

import pytest

from src.rendering import ffmpeg, transitions, video_compose
from src.rendering.video_compose import ClipPart
from tests.video.test_video_compose import FPS, _clip, _compose, _frame

REAL_LOADER = video_compose.load_yaml_config


@pytest.fixture
def pacing(monkeypatch):
    """compose reads configs/video/pacing.yaml; a test sets compose.<key> for itself."""
    def set_compose(**overrides):
        def loader(name):
            config = REAL_LOADER(name)
            if name == "video/pacing.yaml":
                config = {**config, "compose": {**config["compose"], **overrides}}
            return config
        monkeypatch.setattr(video_compose, "load_yaml_config", loader)
    return set_compose


# -- where the fades go --------------------------------------------------------------------------------------

def test_the_fade_length_is_whole_frames_and_zero_means_hard_cuts():
    assert transitions.transition_frames(0.3, 25) == 8
    assert transitions.transition_frames(0.3, 12) == 4
    assert transitions.transition_frames(0.3, 10) == 3
    assert transitions.transition_frames(0, 25) == 0 and transitions.transition_frames(-1, 25) == 0
    assert transitions.transition_frames(0.01, 25) == 2                       # never a one-frame "fade" by accident


def test_a_fade_is_centred_on_the_cut():
    fade = transitions.fades([50, 50], 8)[0]
    assert (fade.frames, fade.lead, fade.tail) == (8, 4, 4)
    odd = transitions.fades([50, 50], 7)[0]
    assert (odd.lead, odd.tail) == (3, 4) and odd.lead + odd.tail == 7


def test_each_clip_runs_past_its_shot_by_the_halves_of_the_fades_at_its_ends():
    assert transitions.margins([50, 60, 40], 8) == [(0, 4), (4, 4), (4, 0)]        # no head on the first, no tail on the last
    assert transitions.margins([50], 8) == [(0, 0)]
    assert transitions.margins([50, 60], 0) == [(0, 0), (0, 0)]


def test_a_fade_never_takes_more_than_half_of_a_short_shot():
    fades = transitions.fades([50, 6, 50], 8)
    assert [f.frames for f in fades] == [3, 3]
    assert transitions.fades([50, 1], 8)[0].frames == 1


def test_the_borrowed_frames_of_a_shot_never_overlap():
    frames = [50, 6, 12, 40]
    found, margin = transitions.fades(frames, 8), transitions.margins(frames, 8)
    for i, shot in enumerate(frames):
        head, tail = margin[i]
        assert head + tail <= shot * 1.01 + 1


# -- in the picture --------------------------------------------------------------------------------------------

def _three(tmp_path, with_margins=True):
    frames = [20, 30, 20]
    margins = transitions.margins(frames, 3) if with_margins else [(0, 0)] * 3
    paths = [_clip(tmp_path / f"{n}.mp4", (f + h + t) / FPS, colour)
             for n, (f, (h, t), colour) in zip("abc", zip(frames, margins, ("red", "green", "blue")))]
    return [ClipPart(p, f) for p, f in zip(paths, frames)]


def test_a_crossfade_blends_the_two_shots_at_the_cut_and_the_video_keeps_its_length(tmp_path, pacing):
    pacing(transition_s=0.3)
    result = _compose(tmp_path, clips=_three(tmp_path), overlays=[], voices=[])
    info = ffmpeg.probe(result.video_path)
    assert info.duration_s == pytest.approx(7.0, abs=0.15)                     # 20 + 30 + 20 frames at 10 fps, as before

    before, mid, after = (_frame(result.video_path, t, tmp_path, f"f{t}").getpixel((135, 240)) for t in (1.5, 2.0, 2.6))
    assert before[0] > 200 and before[1] < 20                                  # red
    assert mid[0] > 40 and mid[1] > 20                                         # both red and green: a blend
    assert after[1] > 100 and after[0] < 20                                    # green


def test_the_blend_is_centred_on_the_cut_not_after_it(tmp_path, pacing):
    pacing(transition_s=0.3)
    result = _compose(tmp_path, clips=_three(tmp_path), overlays=[], voices=[])
    start = _frame(result.video_path, 1.85, tmp_path, "early").getpixel((135, 240))      # before the fade begins (1.9 s)
    late = _frame(result.video_path, 2.3, tmp_path, "late").getpixel((135, 240))         # after it ends (2.2 s)
    assert start[1] < 15 and late[0] < 15


def test_with_the_fade_off_the_cut_is_hard_and_the_length_is_the_same(tmp_path, pacing):
    pacing(transition_s=0)
    result = _compose(tmp_path, clips=_three(tmp_path, with_margins=False), overlays=[], voices=[])
    assert ffmpeg.probe(result.video_path).duration_s == pytest.approx(7.0, abs=0.15)
    before = _frame(result.video_path, 1.85, tmp_path, "b").getpixel((135, 240))
    after = _frame(result.video_path, 2.05, tmp_path, "a").getpixel((135, 240))
    assert before[0] > 200 and before[1] < 20 and after[1] > 100 and after[0] < 20


def test_a_single_clip_has_nothing_to_fade_into(tmp_path, pacing):
    pacing(transition_s=0.3)
    result = _compose(tmp_path, clips=[ClipPart(_clip(tmp_path / "a.mp4", 2, "red"), 20)], overlays=[], voices=[])
    assert ffmpeg.probe(result.video_path).duration_s == pytest.approx(2.0, abs=0.15)


def test_clips_that_were_not_made_longer_still_compose_to_the_right_length(tmp_path, pacing):
    # a clip with no margin is held on its last frame for the fade; the length is the shots' own
    pacing(transition_s=0.3)
    result = _compose(tmp_path, clips=_three(tmp_path, with_margins=False), overlays=[], voices=[])
    assert ffmpeg.probe(result.video_path).duration_s == pytest.approx(7.0, abs=0.15)


def test_the_configured_fade_is_in_pacing_yaml():
    assert REAL_LOADER("video/pacing.yaml")["compose"]["transition_s"] == 0.3
