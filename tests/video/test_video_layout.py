"""Layout: a photograph fills the frame only while that does not enlarge its pixels past the limit; otherwise it is fitted."""
from __future__ import annotations

import pytest
from PIL import Image

from src.agents.video import layout, motion
from src.core.config import load_yaml_config
from src.data.video_schemas import BrandKit, Shot, Storyboard, VideoBrief
from src.models.video_gen import VideoService, fit
from src.models.video_gen.kenburns import KenBurnsProvider, peak_zoom
from src.rendering import ffmpeg
from tests.video.fakes import make_photo

BRAND = BrandKit(name="Aurelia", palette={"dark": "#101018", "accent": "#C0A060"})
CONFIG = {"fill_max_upscale": 1.3, "fit": {"dim": 0.6, "blur_px": 30, "centre_y": 0.47, "zoom": 0.06}}


# -- the decision --------------------------------------------------------------------------------------------

def test_the_limit_is_in_styles_yaml():
    config = load_yaml_config("video/styles.yaml")["layout"]
    assert config["fill_max_upscale"] == 1.3
    assert set(config["fit"]) == {"dim", "blur_px", "centre_y", "zoom"}


def test_peak_zoom_is_the_closest_the_camera_gets():
    assert peak_zoom("still") == 1.0
    assert peak_zoom("push_in") == peak_zoom("pull_out") == pytest.approx(1.18)
    assert all(peak_zoom(c) == pytest.approx(1.15) for c in ("pan_left", "pan_right", "tilt_up", "tilt_down"))


def test_the_fill_peak_is_the_frame_over_the_window_times_the_closest_zoom():
    # a 1568x1059 photograph in 1080x1920: a 596 px wide window shown 1.81 times larger, 1.18 closer at the end of a push
    assert layout.fill_peak(1568, 1059, 1080, 1920, "50% 50%", "push_in") == pytest.approx(1080 / 596 * 1.18, abs=0.01)
    assert layout.fill_peak(1568, 1059, 1080, 1920, "50% 50%", "still") == pytest.approx(1080 / 596, abs=0.01)


@pytest.mark.parametrize("size, camera, expected", [
    ((2919, 1050), "push_in", "fit"),         # a panorama: a 591 px window
    ((1568, 1059), "pan_left", "fit"),        # an ordinary landscape photograph in a tall frame
    ((1576, 2118), "pan_right", "fill"),      # a tall photograph: 1191 px of width for a 1080 px frame
    ((1100, 1950), "push_in", "fill"),        # barely smaller than the frame
])
def test_a_photograph_fills_only_when_it_stays_under_the_limit(size, camera, expected):
    chosen = layout.choose(*size, 1080, 1920, "50% 50%", camera, CONFIG)
    assert chosen.layout == expected
    if expected == "fill":
        assert chosen.peak_upscale == chosen.fill_peak <= 1.3
    else:
        assert chosen.fill_peak > 1.3 and chosen.peak_upscale < chosen.fill_peak


def test_the_limit_is_inclusive_and_configurable():
    peak = layout.fill_peak(1568, 1059, 1080, 1920, "50% 50%", "still")
    assert layout.choose(1568, 1059, 1080, 1920, "50% 50%", "still", {**CONFIG, "fill_max_upscale": peak}).layout == "fill"
    assert layout.choose(1568, 1059, 1080, 1920, "50% 50%", "still", {**CONFIG, "fill_max_upscale": peak - 0.01}).layout == "fit"


def test_a_wide_frame_fills_from_a_wide_photograph():
    assert layout.choose(2000, 1125, 1920, 1080, "50% 50%", "push_in", CONFIG).layout == "fill"
    assert layout.choose(1568, 1059, 1920, 1080, "50% 50%", "push_in", CONFIG).layout == "fit"      # 1.44x: too soft


def test_a_fit_shot_enlarges_far_less_than_a_fill_shot_would():
    chosen = layout.choose(1568, 1059, 1080, 1920, "50% 50%", "push_in", CONFIG)
    assert chosen.fill_peak == pytest.approx(2.14, abs=0.01)
    assert chosen.peak_upscale == pytest.approx(1080 * 1.06 / 1568, abs=0.001) and chosen.peak_upscale < 1.0


def test_a_small_photograph_still_enlarges_in_fit_and_the_peak_says_so():
    assert layout.choose(700, 500, 1080, 1920, "50% 50%", "push_in", CONFIG).peak_upscale == pytest.approx(1080 * 1.06 / 700, abs=0.001)


def test_a_photograph_too_tall_for_the_frame_is_fitted_by_its_height():
    assert fit.base_width(700, 1500, 1080, 1920) == pytest.approx(1920 * 700 / 1500)
    assert fit.base_width(1568, 1059, 1080, 1920) == 1080


# -- the fit frame -------------------------------------------------------------------------------------------------

def _bands(path, size=(1600, 900)):
    """Red on the left, blue on the right, a green stripe across the middle."""
    image = Image.new("RGB", size, (220, 30, 30))
    image.paste((30, 30, 220), (size[0] // 2, 0, size[0], size[1]))
    image.paste((30, 200, 30), (0, size[1] // 2 - 60, size[0], size[1] // 2 + 60))
    image.save(path)
    return path


def _frame(tmp_path, camera="still", progress=0.0, size=(1600, 900)):
    image = Image.open(_bands(tmp_path / "b.png", size)).convert("RGB")
    bg = fit.background(image, "50% 50%", 270, 480, CONFIG["fit"]["dim"], 8)
    return fit.compose_frame(bg, image, 270, 480, 0.47, camera, 0.06, progress), bg


def test_the_whole_photograph_is_laid_across_the_frame_width(tmp_path):
    frame, _ = _frame(tmp_path)
    row = round(480 * 0.47)
    assert frame.getpixel((0, row))[1] > 150 and frame.getpixel((269, row))[1] > 150      # the green stripe reaches both edges
    left, right = frame.getpixel((3, row - 40)), frame.getpixel((266, row - 40))
    assert left[0] > 150 and right[2] > 150                                                # red left, blue right: uncropped, unmirrored
    photo_height = round(270 * 900 / 1600)
    assert frame.getpixel((135, row + photo_height // 2 + 12))[1] < 120                    # below the photograph: the dark background


def test_the_background_is_a_darkened_blurred_copy_of_the_photograph(tmp_path):
    frame, bg = _frame(tmp_path)
    source = Image.open(tmp_path / "b.png").convert("RGB")
    assert bg.size == (270, 480)
    mean = lambda im: sum(sum(p) for p in im.getdata()) / (3 * im.width * im.height)       # noqa: E731
    assert mean(bg) < 0.75 * mean(source.resize((270, 480)))
    top_left, top_right = bg.getpixel((2, 2)), bg.getpixel((267, 2))
    assert top_left[0] > top_left[2] and top_right[2] > top_right[0]                       # still the photograph's colours
    # blurred: the hard red/blue edge in the middle is soft across a few pixels
    assert 40 < bg.getpixel((135, 2))[0] < 160 or abs(bg.getpixel((130, 2))[0] - bg.getpixel((140, 2))[0]) < 120


def test_a_push_in_enlarges_the_photograph_only_never_the_background(tmp_path):
    start, bg = _frame(tmp_path, "push_in", 0.0)
    end, _ = _frame(tmp_path, "push_in", 1.0)
    assert start.getpixel((135, 5)) == end.getpixel((135, 5)) == bg.getpixel((135, 5))     # the background does not move
    row = round(480 * 0.47)
    stripe_height = lambda im: sum(1 for y in range(480) if im.getpixel((135, y))[1] > 150)  # noqa: E731
    assert stripe_height(end) > stripe_height(start)                                       # the photograph is larger


def test_a_pan_slides_the_photograph_without_changing_its_size(tmp_path):
    a, _ = _frame(tmp_path, "pan_right", 0.0)
    b, _ = _frame(tmp_path, "pan_right", 1.0)
    row = round(480 * 0.47) - 30
    edge = lambda im: next(x for x in range(270) if im.getpixel((x, row))[2] > 150)         # where red turns to blue  # noqa: E731
    assert edge(a) > edge(b)                                                                # pan_right: the picture moves left


def test_a_still_does_not_move(tmp_path):
    a, _ = _frame(tmp_path, "still", 0.0)
    b, _ = _frame(tmp_path, "still", 1.0)
    assert a.tobytes() == b.tobytes()


def test_a_fit_clip_is_an_mp4_of_the_asked_size_and_length_with_its_first_frame_saved(tmp_path):
    source = make_photo(tmp_path / "wide.jpg", (1600, 900), seed=4)
    clip = fit.render(source, "push_in", 1.0, 270, 480, 10, tmp_path / "clip.mp4", keyframe_path=tmp_path / "key.png")
    info = ffmpeg.probe(clip.path)
    assert (info.width, info.height, info.vcodec) == (270, 480, "h264")
    assert info.duration_s == pytest.approx(1.0, abs=0.15) and clip.cost_usd == 0
    assert clip.meta == {"camera": "push_in", "layout": "fit"} and Image.open(tmp_path / "key.png").size == (270, 480)


def test_a_fit_clip_from_an_unreadable_picture_is_a_generation_error(tmp_path):
    from src.models.video_gen.base import VideoGenerationError

    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(VideoGenerationError):
        fit.render(bad, "push_in", 1.0, 270, 480, 10, tmp_path / "clip.mp4")


# -- in the motion step ----------------------------------------------------------------------------------------------

def _brief(w=270, h=480, fps=10) -> VideoBrief:
    return VideoBrief(property_id="P", name="Aurelia", duration_s=10, width=w, height=h, fps=fps, quality="draft")


def _board(tmp_path, sizes) -> Storyboard:
    shots = [Shot(index=i, source_path=str(make_photo(tmp_path / "photos" / f"p{i}.jpg", size, seed=i)), seconds=1.0,
                  camera="push_in", focal="50% 50%") for i, size in enumerate(sizes)]
    shots.append(Shot(index=len(shots), kind="card", camera="still", seconds=1.0))
    return Storyboard(shots=shots)


def _service(tmp_path) -> VideoService:
    return VideoService([KenBurnsProvider()], tmp_path / "video_cache")


def test_every_clip_records_its_layout_and_peak_upscale_and_the_shot_carries_them_too(tmp_path):
    board = _board(tmp_path, [(1600, 900), (1100, 1950)])
    result = motion.run(board, _brief(), BRAND, tmp_path / "out", _service(tmp_path))

    wide, tall, card = result.clips
    assert (wide.layout, tall.layout, card.layout) == ("fit", "fill", "")
    assert wide.peak_upscale == pytest.approx(1080 * 1.06 / 1600, abs=0.001)
    assert tall.peak_upscale == pytest.approx(1080 / 1097 * 1.18, abs=0.01) and tall.peak_upscale <= 1.3
    assert [(s.layout, s.peak_upscale) for s in result.storyboard.shots[:2]] == [
        ("fit", wide.peak_upscale), ("fill", tall.peak_upscale)]
    assert all(not s.layout for s in board.shots)                      # the plan that was handed in is left as it was
    assert any(n.startswith("Shot 1: laid out as fit") and "limit 1.3x" in n for n in result.notes)
    assert not any(n.startswith("Shot 2: laid out as fit") for n in result.notes)
    assert set(result.keyframes) == {0, 1} and all(ffmpeg.probe(c.path).width == 270 for c in result.clips)


def test_the_layout_is_worked_out_for_the_final_frame_even_in_a_draft(tmp_path):
    # a 540x960 draft would read a 1600x900 photograph as barely enlarged; the final 1080x1920 video would not
    result = motion.run(_board(tmp_path, [(1600, 900)]), _brief(540, 960), BRAND, tmp_path / "out", _service(tmp_path))
    assert result.clips[0].layout == "fit"


def test_a_stricter_or_looser_limit_changes_the_layout(tmp_path, monkeypatch):
    board = _board(tmp_path, [(1600, 900)])
    monkeypatch.setattr(layout, "settings", lambda: {**CONFIG, "fill_max_upscale": 5.0})
    assert motion.run(board, _brief(), BRAND, tmp_path / "loose", _service(tmp_path)).clips[0].layout == "fill"
    monkeypatch.setattr(layout, "settings", lambda: {**CONFIG, "fill_max_upscale": 0.5})
    assert motion.run(board, _brief(), BRAND, tmp_path / "strict", _service(tmp_path)).clips[0].layout == "fit"


def test_a_fit_shot_never_goes_to_a_video_model(tmp_path):
    from tests.video.fakes import ScriptedVideoProvider, paid_spec

    paid = ScriptedVideoProvider("paid", spec=paid_spec(key_env=None))
    service = VideoService([paid, KenBurnsProvider()], tmp_path / "video_cache")
    result = motion.run(_board(tmp_path, [(1600, 900), (1100, 1950)]), _brief(), BRAND, tmp_path / "out", service)
    assert [c.provider for c in result.clips[:2]] == ["kenburns", "paid"] and len(paid.calls) == 1
