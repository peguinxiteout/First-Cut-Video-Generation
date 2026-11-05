"""Node 7: keyframes, cards and clips."""
from __future__ import annotations

import pytest
from PIL import Image

from src.agents.video import motion
from src.data.video_schemas import BrandKit, Shot, Storyboard, VideoBrief
from src.models.video_gen import VideoGenerationTransportError, VideoService
from src.models.video_gen.kenburns import KenBurnsProvider
from src.rendering import ffmpeg, transitions
from src.rendering.video_compose import shot_frames
from tests.video.fakes import ScriptedVideoProvider, make_photo, paid_spec

BRAND = BrandKit(name="Aurelia", palette={"dark": "#101018", "accent": "#C0A060"})


@pytest.fixture(autouse=True)
def _photographs_fill(monkeypatch):
    """These tests are about the keyframe and the model chain, which only a shot laid out as "fill"
    goes through; the small test photographs would all be fitted. tests/video/test_video_layout.py
    covers the choice."""
    monkeypatch.setattr("src.agents.video.layout.settings", lambda: {
        "fill_max_upscale": 99.0, "fit": {"dim": 0.6, "blur_px": 30, "centre_y": 0.47, "zoom": 0.06}})


def _brief(w=270, h=480, fps=10) -> VideoBrief:
    return VideoBrief(property_id="P", name="Aurelia", duration_s=10, width=w, height=h, fps=fps, quality="draft")


def _halves(path, size=(1600, 900)):
    """Left half red, right half blue."""
    image = Image.new("RGB", size, (200, 30, 30))
    image.paste((30, 30, 200), (size[0] // 2, 0, size[0], size[1]))
    image.save(path)
    return path


# -- the crop ------------------------------------------------------------------------------------

def test_the_crop_is_the_largest_window_of_the_videos_shape():
    left, top, right, bottom = motion.crop_box(1600, 900, 9 / 16, "50% 50%")
    assert (right - left, bottom - top) == (506, 900)
    assert left == round(0.5 * (1600 - 506)) and top == 0
    left, top, right, bottom = motion.crop_box(900, 1600, 16 / 9, "50% 50%")           # a tall photo in a wide frame
    assert (right - left, bottom - top) == (900, 506)


@pytest.mark.parametrize("focal, expected_left", [("0% 50%", 0), ("100% 50%", 1094), ("25% 50%", 274)])
def test_the_focal_point_places_the_window_as_object_position_does(focal, expected_left):
    assert motion.crop_box(1600, 900, 9 / 16, focal)[0] == expected_left


def test_the_keyframe_shows_the_part_of_the_picture_the_focal_point_names(tmp_path):
    source = _halves(tmp_path / "halves.png")
    left = motion.make_keyframe(source, "0% 50%", 270, 480, tmp_path / "l.png")
    right = motion.make_keyframe(source, "100% 50%", 270, 480, tmp_path / "r.png")
    red, blue = Image.open(left.path).getpixel((135, 240)), Image.open(right.path).getpixel((135, 240))
    assert red[0] > 150 and red[2] < 80
    assert blue[2] > 150 and blue[0] < 80


def test_a_small_photograph_is_enlarged_to_the_frame_and_the_original_size_is_reported(tmp_path):
    source = make_photo(tmp_path / "small.jpg", (640, 360), seed=3)
    keyframe = motion.make_keyframe(source, "50% 50%", 540, 960, tmp_path / "k.png")
    assert (keyframe.width, keyframe.height) == (540, 960)
    assert keyframe.upscaled_from == (202, 360)
    assert Image.open(keyframe.path).size == (540, 960)


def test_a_photograph_big_enough_is_kept_at_its_own_resolution(tmp_path):
    source = make_photo(tmp_path / "big.jpg", (3200, 1800), seed=4)
    keyframe = motion.make_keyframe(source, "50% 50%", 540, 960, tmp_path / "k.png")
    assert keyframe.upscaled_from is None
    assert (keyframe.width, keyframe.height) == (1012, 1800)
    assert keyframe.width / keyframe.height == pytest.approx(540 / 960, abs=0.005)


def test_a_huge_photograph_is_not_kept_larger_than_twice_the_frame(tmp_path):
    source = make_photo(tmp_path / "huge.jpg", (6000, 3375), seed=5)
    keyframe = motion.make_keyframe(source, "50% 50%", 540, 960, tmp_path / "k.png")
    assert (keyframe.width, keyframe.height) == (1080, 1920)


# -- cards ---------------------------------------------------------------------------------------

def _logo(tmp_path, size=(200, 120)):
    path = tmp_path / "logo.jpg"
    Image.new("RGB", size, (255, 255, 255)).save(path)
    image = Image.open(path)
    image.paste((20, 80, 160), (20, 20, size[0] - 20, size[1] - 20))
    image.save(path)
    return path


def test_a_card_is_the_palette_with_the_logo_on_a_white_plate(tmp_path):
    brand = BrandKit(name="A", palette=BRAND.palette, logo_path=str(_logo(tmp_path)))
    drawn = motion.render_card(brand, 540, 960, tmp_path / "card.png")

    image = Image.open(tmp_path / "card.png").convert("RGB")
    assert drawn is True and image.size == (540, 960)
    assert image.getpixel((2, 2)) == pytest.approx((16, 16, 24), abs=3)                # the palette's dark
    top, foot = image.getpixel((2, 2)), image.getpixel((2, 957))
    assert sum(foot) > sum(top)                                                         # a little lighter towards the foot
    plate = image.getpixel((270 - 100, round(960 * motion.LOGO_CENTRE_Y)))             # inside the plate, beside the mark
    assert plate[0] > 240 and plate[1] > 240 and plate[2] > 240 or plate[2] > 120      # white plate or the logo itself
    assert image.getpixel((135, round(960 * motion.LOGO_CENTRE_Y))) != image.getpixel((135, 700))


def test_a_logo_is_never_enlarged_past_twice_its_size(tmp_path):
    brand = BrandKit(name="A", palette=BRAND.palette, logo_path=str(_logo(tmp_path, (100, 60))))
    motion.render_card(brand, 1080, 1920, tmp_path / "card.png")
    image = Image.open(tmp_path / "card.png").convert("RGB")
    row = [image.getpixel((x, round(1920 * motion.LOGO_CENTRE_Y))) for x in range(1080)]
    white = [i for i, px in enumerate(row) if px[0] > 240 and px[1] > 240 and px[2] > 240]
    blue = [i for i, px in enumerate(row) if px[2] > 120 and px[0] < 60]
    assert blue and (max(blue) - min(blue)) <= 2 * 100


def test_a_card_without_a_logo_is_just_the_palette(tmp_path):
    assert motion.render_card(BRAND, 270, 480, tmp_path / "card.png") is False
    image = Image.open(tmp_path / "card.png").convert("RGB")
    assert not any(image.getpixel((x, y)) == (255, 255, 255) for x in range(0, 270, 9) for y in range(0, 480, 9))


def test_an_unreadable_logo_leaves_the_card_without_one(tmp_path):
    bad = tmp_path / "logo.jpg"
    bad.write_text("not an image", encoding="utf-8")
    brand = BrandKit(name="A", palette=BRAND.palette, logo_path=str(bad))
    assert motion.render_card(brand, 270, 480, tmp_path / "card.png") is False


def test_a_still_clip_has_the_asked_frames(tmp_path):
    png = tmp_path / "card.png"
    motion.render_card(BRAND, 270, 480, png)
    motion.still_clip(png, 1.5, 10, tmp_path / "card.mp4")
    info = ffmpeg.probe(tmp_path / "card.mp4")
    assert (info.width, info.height, info.vcodec, info.pix_fmt, info.fps) == (270, 480, "h264", "yuv420p", 10)
    assert info.duration_s == pytest.approx(1.5, abs=0.1)


# -- the clips -----------------------------------------------------------------------------------

def _board(tmp_path, kinds=("real", "real", "card")) -> Storyboard:
    shots = []
    for i, kind in enumerate(kinds):
        path = str(make_photo(tmp_path / "photos" / f"p{i}.jpg", (1600, 900), seed=10 + i)) if kind == "real" else ""
        shots.append(Shot(index=i, kind=kind, source_path=path, camera="push_in" if kind == "real" else "still",
                          seconds=1.0, focal="50% 50%", motion_prompt="slow push in"))
    return Storyboard(shots=shots)


def _service(tmp_path, *providers) -> VideoService:
    return VideoService([*providers, KenBurnsProvider()], tmp_path / "video_cache")


def test_every_shot_gets_a_clip_of_the_videos_size_and_a_record_of_how(tmp_path):
    result = motion.run(_board(tmp_path), _brief(), BRAND, tmp_path / "out", _service(tmp_path))

    assert [(c.shot_index, c.mode, c.provider, c.ok) for c in result.clips] == [
        (0, "kenburns", "kenburns", True), (1, "kenburns", "kenburns", True), (2, "card", "", True)]
    assert set(result.keyframes) == {0, 1}
    # each clip is its shot plus what its crossfades borrow (0.3 s centred on each cut: 3 frames at 10 fps)
    margins = transitions.margins(shot_frames([1.0, 1.0, 1.0], 10), 3)
    for clip, (head, tail) in zip(result.clips, margins):
        info = ffmpeg.probe(clip.path)
        assert (info.width, info.height, info.vcodec) == (270, 480, "h264")
        assert info.duration_s == pytest.approx((10 + head + tail) / 10, abs=0.15)
    assert all(not c.from_cache and c.cost_usd == 0 for c in result.clips)


def test_the_clip_comes_from_the_keyframe_not_the_original_photograph(tmp_path):
    board = _board(tmp_path)
    paid = ScriptedVideoProvider("paid", spec=paid_spec(key_env=None))
    motion.run(board, _brief(), BRAND, tmp_path / "out", _service(tmp_path, paid))
    request = paid.calls[0]
    assert request.image_path.name == "keyframe_00.png" and request.camera == "push_in"
    assert request.prompt == "slow push in" and (request.width, request.height, request.fps) == (270, 480, 10)
    assert Image.open(request.image_path).size[0] / Image.open(request.image_path).size[1] == pytest.approx(270 / 480, abs=0.01)


def test_a_clip_that_fails_falls_to_pan_and_zoom_for_that_shot_and_the_run_goes_on(tmp_path):
    paid = ScriptedVideoProvider("paid", script=[VideoGenerationTransportError("timed out")])
    result = motion.run(_board(tmp_path), _brief(), BRAND, tmp_path / "out", _service(tmp_path, paid))

    first, second = result.clips[0], result.clips[1]
    assert (first.provider, first.mode, first.ok) == ("kenburns", "kenburns", True)
    assert "paid: timed out" in first.reason
    assert (second.provider, second.mode) == ("paid", "model")                     # tried again on the next shot
    assert second.cost_usd > 0 and second.reason == ""
    assert not result.notes


def test_a_second_run_over_the_same_pictures_is_served_from_the_cache(tmp_path):
    paid = ScriptedVideoProvider("paid")
    board = _board(tmp_path)
    motion.run(board, _brief(), BRAND, tmp_path / "one", _service(tmp_path, paid))
    again = motion.run(board, _brief(), BRAND, tmp_path / "two", _service(tmp_path, paid))
    assert [c.from_cache for c in again.clips if c.mode == "model"] == [True, True]
    assert len(paid.calls) == 2 and all(c.cost_usd == 0 for c in again.clips)


def test_a_photograph_that_cannot_be_cut_becomes_a_title_card_with_the_reason(tmp_path):
    board = _board(tmp_path)
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image", encoding="utf-8")
    board.shots[1].source_path = str(bad)

    result = motion.run(board, _brief(), BRAND, tmp_path / "out", _service(tmp_path))

    clip = result.clips[1]
    assert (clip.mode, clip.ok) == ("card", False) and "UnidentifiedImageError" in clip.reason
    assert any(n.startswith("Shot 2 is a title card") for n in result.notes)
    assert ffmpeg.probe(clip.path).width == 270 and 1 not in result.keyframes


def test_a_generated_shot_is_a_title_card_until_pictures_can_be_made(tmp_path):
    board = _board(tmp_path, ("real", "generated", "card"))
    result = motion.run(board, _brief(), BRAND, tmp_path / "out", _service(tmp_path))
    assert result.clips[1].mode == "card" and "not made yet" in result.clips[1].reason and not result.clips[1].ok


def test_a_small_photograph_is_noted_when_it_was_enlarged(tmp_path):
    board = _board(tmp_path)
    small = make_photo(tmp_path / "photos" / "tiny.jpg", (500, 470), seed=30)       # fills with a 1.02x enlargement
    board.shots[0].source_path = str(small)
    result = motion.run(board, _brief(270, 480), BRAND, tmp_path / "out", _service(tmp_path))
    assert result.clips[0].layout == "fill"
    assert any(n.startswith("Shot 1: the photograph is small (264x470 after cropping)") for n in result.notes)


def test_a_chosen_model_that_cannot_run_is_said_once(tmp_path):
    service = VideoService([KenBurnsProvider()], tmp_path / "c", note="Seedance 2.0 is unavailable (no key)")
    result = motion.run(_board(tmp_path), _brief(), BRAND, tmp_path / "out", service)
    assert result.notes == ["Seedance 2.0 is unavailable (no key)"]
    assert result.clips[0].reason == "Seedance 2.0 is unavailable (no key)"
