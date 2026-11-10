from __future__ import annotations

import pytest

from src.data.video_schemas import CAMERA_MOVES
from src.models.video_gen.base import ImageToVideoRequest, ProviderSpec, VideoGenerationError
from src.models.video_gen.kenburns import Geometry, KenBurnsProvider, window_at
from src.rendering import ffmpeg
from tests.video.fakes import small_request


def _request(image, **overrides) -> ImageToVideoRequest:
    fields = dict(width=270, height=480, fps=12, duration_s=1.0)
    fields.update(overrides)
    return small_request(image, **fields)


@pytest.mark.parametrize("camera", CAMERA_MOVES)
def test_every_camera_move_makes_a_clip_of_the_asked_size_fps_and_length(photo, tmp_path, camera):
    clip = KenBurnsProvider().generate(_request(photo, camera=camera), tmp_path / f"{camera}.mp4")

    info = ffmpeg.probe(clip.path)
    assert (info.width, info.height) == (270, 480)
    assert info.vcodec == "h264" and info.pix_fmt == "yuv420p"
    assert info.fps == 12
    assert info.duration_s == pytest.approx(1.0, abs=0.1)
    assert not info.has_audio
    assert (clip.provider, clip.cost_usd, clip.duration_s) == ("kenburns", 0.0, 1.0)


def test_a_portrait_photograph_works_for_the_horizontal_moves_too(portrait_photo, tmp_path):
    for camera in ("pan_left", "pan_right", "tilt_up"):
        clip = KenBurnsProvider().generate(_request(portrait_photo, camera=camera), tmp_path / f"{camera}.mp4")
        assert ffmpeg.probe(clip.path).duration_s == pytest.approx(1.0, abs=0.1)


def test_the_clip_is_held_to_the_configured_maximum(photo, tmp_path):
    spec = ProviderSpec(id="kenburns", label="Pan-and-zoom", adapter="kenburns", max_duration_s=1.0)
    clip = KenBurnsProvider(spec).generate(_request(photo, duration_s=3.0), tmp_path / "a.mp4")
    assert clip.duration_s == 1.0 and clip.meta["clamped_from_s"] == 3.0
    assert ffmpeg.probe(clip.path).duration_s == pytest.approx(1.0, abs=0.1)


def test_odd_sizes_are_rounded_down_to_even_and_reported(photo, tmp_path):
    clip = KenBurnsProvider().generate(_request(photo, width=271, height=479), tmp_path / "a.mp4")
    assert (clip.width, clip.height) == (270, 478)
    assert (ffmpeg.probe(clip.path).width, ffmpeg.probe(clip.path).height) == (270, 478)


def test_a_missing_picture_and_an_unknown_move_are_clear_errors(photo, tmp_path):
    with pytest.raises(VideoGenerationError, match="does not exist"):
        KenBurnsProvider().generate(_request(tmp_path / "nope.jpg"), tmp_path / "a.mp4")
    with pytest.raises(VideoGenerationError, match="unknown camera move 'whip_pan'"):
        KenBurnsProvider().generate(_request(photo, camera="whip_pan"), tmp_path / "a.mp4")


def test_a_file_that_is_not_a_picture_is_a_clear_error(tmp_path):
    bad = tmp_path / "notes.jpg"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(VideoGenerationError, match="cannot read notes.jpg"):
        KenBurnsProvider().generate(_request(bad), tmp_path / "a.mp4")


# -- the movement itself, without encoding anything ---------------------------------

WIDE = Geometry.of(1600, 900, 1080, 1920, "50% 50%")
TALL = Geometry.of(900, 1600, 1080, 1920, "50% 50%")


def _start_end(camera, geometry=WIDE):
    return window_at(camera, 0.0, geometry), window_at(camera, 1.0, geometry)


def test_pan_left_moves_the_window_left_and_pan_right_right():
    (l0, *_), (l1, *_) = _start_end("pan_left")
    assert l1 < l0
    (l0, *_), (l1, *_) = _start_end("pan_right")
    assert l1 > l0


def test_tilt_up_moves_the_window_up_and_tilt_down_down():
    (_, t0, *_), (_, t1, *_) = _start_end("tilt_up", TALL)
    assert t1 < t0
    (_, t0, *_), (_, t1, *_) = _start_end("tilt_down", TALL)
    assert t1 > t0


def test_push_in_shrinks_the_window_and_pull_out_grows_it():
    (l0, t0, r0, b0), (l1, t1, r1, b1) = _start_end("push_in")
    assert (r1 - l1) < (r0 - l0) and (b1 - t1) < (b0 - t0)
    (l0, t0, r0, b0), (l1, t1, r1, b1) = _start_end("pull_out")
    assert (r1 - l1) > (r0 - l0)


def test_a_still_does_not_move():
    start, end = _start_end("still")
    assert start == pytest.approx(end)


@pytest.mark.parametrize("focal", ["0% 0%", "100% 100%", "50% 50%", "3% 97%", "garbage", ""])
@pytest.mark.parametrize("camera", CAMERA_MOVES)
@pytest.mark.parametrize("size", [(1600, 900), (900, 1600), (1000, 1000)])
def test_the_window_always_stays_inside_the_picture_and_keeps_the_output_shape(camera, focal, size):
    g = Geometry.of(*size, 1080, 1920, focal)
    for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
        left, top, right, bottom = window_at(camera, progress, g)
        assert -1e-6 <= left < right <= g.src_w + 1e-6
        assert -1e-6 <= top < bottom <= g.src_h + 1e-6
        assert (right - left) / (bottom - top) == pytest.approx(1080 / 1920, rel=1e-6)


def test_the_window_starts_around_the_focal_point():
    g = Geometry.of(1600, 900, 1080, 1920, "10% 50%")
    left, _, right, _ = window_at("still", 0.0, g)
    assert (left + right) / 2 < 400
