"""Keyframes for an image-to-video model from photographs of any shape: crop,
extend (with a fake outpaint) and travel, and the mode each is recommended."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from src.agents.video import keyframe
from src.models.video_gen.base import VideoGenerationError
from tests.video.fakes import make_photo

SIZE = (720, 1280)


def _subject(path, size, box, seed=0):
    """A flat picture with all its detail inside `box` (shares of the picture): a subject that does or
    does not fit a crop."""
    w, h = size
    image = Image.new("RGB", size, (120, 160, 200))
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = (round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h))
    for _ in range(400):
        x, y = int(rng.integers(x0, max(x0 + 1, x1 - 8))), int(rng.integers(y0, max(y0 + 1, y1 - 8)))
        draw.rectangle((x, y, x + 6, y + 6), fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    return path


def keep_band(image: Image.Image, expand: dict) -> Image.Image:
    """A fake outpaint that leaves the picture as it is and fills the margins with noise."""
    w = image.width + expand["expand_left"] + expand["expand_right"]
    h = image.height + expand["expand_top"] + expand["expand_bottom"]
    noise = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
    canvas = Image.fromarray(noise)
    canvas.paste(image, (expand["expand_left"], expand["expand_top"]))
    return canvas


# ---------------------------------------------------------------------------
# crop

@pytest.mark.parametrize("size", [(1600, 900), (900, 1600), (1000, 1000), (160, 90), (720, 1280)])
def test_a_crop_is_the_models_size_whatever_the_shape(tmp_path, size):
    photo = make_photo(tmp_path / f"p{size[0]}x{size[1]}.jpg", size, seed=size[0])
    kf = keyframe.crop(photo, tmp_path / "kf.jpg")
    with Image.open(kf.path) as im:
        assert im.size == SIZE
    left, top, right, bottom = kf.windows[0]
    assert abs((right - left) / (bottom - top) - 720 / 1280) < 0.01
    assert kf.enlargement == pytest.approx(max(720 / (right - left), 1280 / (bottom - top)), abs=1e-3)
    assert not kf.generated


def test_a_tiny_photograph_reports_a_large_enlargement(tmp_path):
    kf = keyframe.crop(make_photo(tmp_path / "tiny.jpg", (160, 90)), tmp_path / "kf.jpg")
    assert kf.enlargement == pytest.approx(1280 / 90, abs=0.01)


def test_the_crop_follows_the_subject(tmp_path):
    photo = _subject(tmp_path / "right.png", (1600, 900), (0.72, 0.1, 0.95, 0.9))
    kf = keyframe.crop(photo, tmp_path / "kf.jpg")
    left, _, right, _ = kf.windows[0]
    assert left > 800 and right <= 1600


# ---------------------------------------------------------------------------
# extend

@pytest.mark.parametrize("size", [(1600, 900), (500, 1600), (1000, 1000), (160, 90)])
def test_extend_keeps_every_pixel_of_the_photograph_in_the_middle(tmp_path, size):
    photo = make_photo(tmp_path / "p.png", size, seed=4)
    kf = keyframe.extend(photo, tmp_path / "kf.png", keep_band)
    plan = keyframe.extend_plan(*size)
    left, top, right, bottom = kf.band
    assert (right - left, bottom - top) == plan["band_size"]
    assert kf.generated and kf.band_check["verdict"] == "exact" and kf.band_check["mean_diff_after"] == 0.0
    with Image.open(photo) as im:
        want = np.asarray(im.convert("RGB").resize(plan["band_size"], Image.LANCZOS), dtype=np.int16)
    with Image.open(kf.path) as im:
        assert im.size == SIZE
        got = np.asarray(im.convert("RGB"), dtype=np.int16)[top:bottom, left:right]
    assert np.abs(got - want).mean() < 1.5          # the file is a JPEG at quality 95


def test_a_wide_photograph_is_extended_top_and_bottom_a_tall_one_at_the_sides():
    wide, tall, square = keyframe.extend_plan(1600, 900), keyframe.extend_plan(400, 1600), keyframe.extend_plan(800, 800)
    assert wide["expand"]["expand_left"] == wide["expand"]["expand_right"] == 0
    assert wide["expand"]["expand_top"] + wide["expand"]["expand_bottom"] == 1280 - 405
    assert tall["expand"]["expand_top"] == tall["expand"]["expand_bottom"] == 0
    assert tall["expand"]["expand_left"] + tall["expand"]["expand_right"] == 720 - 320
    assert square["band_size"] == (720, 720) and square["expand"]["expand_top"] == 280


@pytest.mark.parametrize("size,shift", [((1600, 900), (0, -3)), ((500, 1600), (4, 0))])
def test_a_band_moved_a_few_pixels_is_found_and_the_original_put_back(tmp_path, size, shift):
    photo = make_photo(tmp_path / "p.png", size, seed=5)

    def shifted(image, expand):
        canvas = keep_band(image, expand)
        out = Image.fromarray(np.random.default_rng(2).integers(0, 255, (canvas.height, canvas.width, 3),
                                                                 dtype=np.uint8))
        out.paste(canvas, shift)
        return out
    kf = keyframe.extend(photo, tmp_path / "kf.png", shifted)
    assert kf.band_check["offset"] == list(shift) and kf.band_check["verdict"] == "exact"
    plan = keyframe.extend_plan(*size)
    assert (kf.band[0], kf.band[1]) == (plan["left"] + shift[0], plan["top"] + shift[1])


def test_a_slightly_changed_band_is_near_and_restored(tmp_path):
    photo = make_photo(tmp_path / "p.png", (1600, 900), seed=6)
    kf = keyframe.extend(photo, tmp_path / "kf.png",
                         lambda im, ex: keep_band(im.filter(ImageFilter.GaussianBlur(0.8)), ex))
    assert kf.band_check["verdict"] in ("exact", "near") and kf.band_check["mean_diff_after"] == 0.0


def test_an_outpaint_that_redraws_the_photograph_is_refused(tmp_path):
    photo = make_photo(tmp_path / "p.png", (1600, 900), seed=7)
    other = Image.open(make_photo(tmp_path / "o.png", (1600, 900), seed=99)).convert("RGB")
    with pytest.raises(VideoGenerationError, match="changed the photograph"):
        keyframe.extend(photo, tmp_path / "kf.png", lambda im, ex: keep_band(other.resize(im.size), ex))


def test_an_answer_of_another_size_is_resized_and_said(tmp_path):
    photo = make_photo(tmp_path / "p.png", (1600, 900), seed=8)
    kf = keyframe.extend(photo, tmp_path / "kf.png", lambda im, ex: keep_band(im, ex).resize((1440, 2560)))
    assert any("resized" in n for n in kf.notes) and kf.band_check["verdict"] in ("exact", "near")


def test_a_photograph_already_the_videos_shape_needs_no_outpaint(tmp_path):
    photo = make_photo(tmp_path / "p.png", (720, 1280))

    def never(*_):
        raise AssertionError("no outpaint")
    kf = keyframe.extend(photo, tmp_path / "kf.png", never)
    assert not kf.generated and kf.cost_usd == 0.0


def test_the_outpaint_price_follows_the_model_page():
    # 720x381 in, 720x1280 out: 0.27 + 0.92 MP, one extra megapixel rounded up
    assert keyframe.outpaint_price((720, 381), (720, 1280)) == pytest.approx(0.045)
    assert keyframe.outpaint_price((200, 200), (400, 400)) == pytest.approx(0.03)
    assert keyframe.outpaint_price((1080, 571), (1080, 1920)) == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# travel

@pytest.mark.parametrize("size", [(1920, 1015), (1600, 900), (500, 1600)])
def test_travel_gives_two_real_windows_inside_the_photograph(tmp_path, size):
    photo = make_photo(tmp_path / "p.png", size, seed=10)
    kf = keyframe.travel(photo, tmp_path / "a.jpg", tmp_path / "b.jpg")
    first, last = kf.windows
    for box in (first, last):
        assert box[0] >= 0 and box[1] >= 0 and box[2] <= size[0] and box[3] <= size[1]
    for path in (kf.path, kf.end_path):
        with Image.open(path) as im:
            assert im.size == SIZE
    wide = size[0] / size[1] > 720 / 1280
    shift = (last[0] - first[0]) if wide else (last[1] - first[1])
    window = (first[2] - first[0]) if wide else (first[3] - first[1])
    assert 0.25 * window - 1 <= shift <= window + 1


def test_travel_crosses_a_wide_subject():
    profile = np.zeros(200)
    profile[20:180] = 1.0                           # detail across 80% of the width
    first, last = keyframe.travel_windows(1920, 1015, profile=profile)
    window = first[2] - first[0]
    assert last[0] - first[0] == pytest.approx(window, abs=2)        # capped at one window
    assert first[0] < 960 < last[2]


# ---------------------------------------------------------------------------
# recommend

def test_a_photograph_of_about_the_videos_shape_is_cropped(tmp_path):
    rec = keyframe.recommend(make_photo(tmp_path / "p.png", (760, 1280)))
    assert rec.mode == "crop" and "nothing is lost" in rec.reasons[-1]


def test_a_wide_photograph_with_its_subject_in_one_window_is_cropped(tmp_path):
    rec = keyframe.recommend(_subject(tmp_path / "p.png", (1600, 900), (0.40, 0.1, 0.58, 0.9)))
    assert rec.mode == "crop" and rec.subject_fits and rec.crop_detail >= 0.6


def test_a_wide_photograph_with_its_subject_across_it_travels(tmp_path):
    rec = keyframe.recommend(_subject(tmp_path / "p.png", (1920, 1015), (0.05, 0.1, 0.95, 0.9)))
    assert rec.mode == "travel" and not rec.subject_fits
    assert rec.crop_keep == pytest.approx(0.30, abs=0.01)


def test_a_photograph_a_little_wider_than_the_video_with_its_subject_at_both_sides_is_extended(tmp_path):
    path = _subject(tmp_path / "p.png", (750, 1000), (0.0, 0.1, 0.18, 0.9), seed=1)
    right = _subject(tmp_path / "r.png", (750, 1000), (0.82, 0.1, 1.0, 0.9), seed=2)
    with Image.open(path) as a, Image.open(right) as b:
        both = Image.fromarray(np.where(np.asarray(b) != np.asarray(Image.new("RGB", b.size, (120, 160, 200))),
                                        np.asarray(b), np.asarray(a)))
    both.save(path)
    rec = keyframe.recommend(path)
    assert rec.mode == "extend" and rec.crop_detail < 0.6


def test_a_square_photograph_with_its_subject_across_it_travels_rather_than_being_mostly_generated(tmp_path):
    rec = keyframe.recommend(_subject(tmp_path / "p.png", (1000, 1000), (0.0, 0.1, 1.0, 0.9)))
    assert rec.mode == "travel"


def test_a_tiny_photograph_is_said_to_look_soft(tmp_path):
    rec = keyframe.recommend(make_photo(tmp_path / "p.png", (160, 90)))
    assert any("soft" in r for r in rec.reasons)
