"""The truth check on synthetic clips: camera motion over the keyframe passes, a
different picture fails, a reshaped building scores lower, generated margins of
an extended keyframe are not scored, and a travel shot is matched across the
wide photograph."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.agents.video import motion_check
from src.rendering import ffmpeg


def facade(size=(360, 640), seed=0) -> Image.Image:
    """A picture with a building's kind of detail: a block of windows, a roof line, planting."""
    w, h = size
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", size, (150, 190, 230))
    draw = ImageDraw.Draw(image)
    draw.rectangle((w * 0.2, h * 0.15, w * 0.8, h * 0.85), fill=(200, 196, 188))
    for row in range(14):
        for col in range(6):
            x, y = w * 0.24 + col * w * 0.093, h * 0.19 + row * h * 0.047
            draw.rectangle((x, y, x + w * 0.05, y + h * 0.03), fill=(60, 70, 90))
    for _ in range(40):
        x, y, r = rng.integers(0, w), rng.integers(int(h * 0.8), h), rng.integers(8, 30)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(40, int(rng.integers(90, 160)), 50))
    return image


def zoom(image: Image.Image, z: float, cx: float = 0.5, cy: float = 0.5, size=None) -> np.ndarray:
    W, H = image.size
    w, h = W / z, H / z
    left = min(max(cx * W - w / 2, 0), W - w)
    top = min(max(cy * H - h / 2, 0), H - h)
    return np.asarray(image.resize(size or (W, H), Image.LANCZOS, box=(left, top, left + w, top + h)))


def reshape(frame: np.ndarray) -> np.ndarray:
    """The upper half of the building stretched 15% taller: a floor added."""
    image = Image.fromarray(frame)
    W, H = image.size
    upper = image.crop((0, 0, W, H // 2)).resize((W, int(H / 2 * 1.15)), Image.LANCZOS)
    out = image.copy()
    out.paste(upper.crop((0, upper.height - H // 2, W, upper.height)), (0, 0))
    return np.asarray(out)


@pytest.fixture(scope="module")
def keyframe_image():
    return facade()


def test_ssim_and_the_hash_of_one_picture_and_of_two():
    a = motion_check.gray(facade(seed=1))
    assert motion_check.ssim(a, a) == pytest.approx(1.0)
    assert motion_check.hamming(motion_check.phash(a), motion_check.phash(a)) == 0
    assert motion_check.ssim(a, a + 3) > 0.99
    assert motion_check.ssim(a, np.flipud(a)) < motion_check.ssim(a, a + 3) - 0.2
    assert motion_check.hamming(motion_check.phash(a), motion_check.phash(np.flipud(a))) > 10


def test_a_push_in_over_the_keyframe_passes_and_the_zoom_is_found(keyframe_image):
    frames = [zoom(keyframe_image, z) for z in (1.0, 1.08, 1.16)]
    truth = motion_check.check_frames(frames, [0.0, 1.0, 2.0], np.asarray(keyframe_image))
    assert truth.verdict == "pass", truth.reasons
    assert truth.first_ssim > 0.95 and truth.first_phash <= 2
    assert [f.zoom for f in truth.frames] == pytest.approx([1.0, 1.08, 1.16], abs=0.03)
    assert min(f.mean for f in truth.frames) > 0.85 and truth.score > 85


def test_another_picture_fails(keyframe_image):
    other = np.asarray(Image.open(_blobs()).convert("RGB").resize(keyframe_image.size))
    truth = motion_check.check_frames([other] * 3, [0.0, 1.0, 2.0], np.asarray(keyframe_image))
    assert truth.verdict == "fail"
    assert truth.first_ssim < 0.65 and truth.score < 50


def test_a_reshaped_building_scores_lower_than_camera_motion(keyframe_image):
    clean = [zoom(keyframe_image, z) for z in (1.0, 1.08, 1.16)]
    changed = [clean[0]] + [reshape(f) for f in clean[1:]]
    a = motion_check.check_frames(clean, [0.0, 1.0, 2.0], np.asarray(keyframe_image))
    b = motion_check.check_frames(changed, [0.0, 1.0, 2.0], np.asarray(keyframe_image))
    assert b.score < a.score - 5
    assert min(f.p10 for f in b.frames) < min(f.p10 for f in a.frames) - 0.2
    assert b.verdict != "pass"


def test_only_the_photographs_band_of_an_extended_keyframe_is_scored(keyframe_image):
    w, h = keyframe_image.size
    band = (0, h // 4, w, 3 * h // 4)
    key = np.asarray(keyframe_image).copy()
    frames = []
    for seed in range(3):
        f = key.copy()
        noise = np.random.default_rng(seed).integers(0, 255, f.shape, dtype=np.uint8)
        f[:band[1]], f[band[3]:] = noise[:band[1]], noise[band[3]:]       # the generated margins change freely
        frames.append(f)
    scored = motion_check.check_frames(frames, [0.0, 1.0, 2.0], key, band=band)
    whole = motion_check.check_frames(frames, [0.0, 1.0, 2.0], key)
    assert scored.region == "band" and scored.verdict == "pass", scored.reasons
    assert scored.score > whole.score


def test_a_travel_shot_is_matched_across_the_wide_photograph():
    wide = facade((1280, 640), seed=3)
    win = (360, 640)
    # a 3 s travel of about one window, sampled every half second: 60 px a sample
    frames = [zoom(wide.crop((x, 0, x + 360, 640)), 1.0) for x in (100, 160, 220, 280)]
    truth = motion_check.check_frames(frames, [0.0, 0.5, 1.0, 1.5], frames[0], reference=np.asarray(wide),
                                      start_box=(100, 0, 460, 640))
    assert truth.verdict == "pass", truth.reasons
    xs = [f.x for f in truth.frames]
    assert xs == sorted(xs) and xs[0] == pytest.approx((100 + win[0] / 2) / 1280, abs=0.03)


def test_a_clip_file_is_read_and_sampled_every_half_second(tmp_path, keyframe_image):
    key = tmp_path / "kf.png"
    keyframe_image.save(key)
    frames = [zoom(keyframe_image, 1.0 + 0.1 * i / 19, size=(180, 320)) for i in range(20)]
    clip = tmp_path / "clip.mp4"
    ffmpeg.encode_frames((f.tobytes() for f in frames), clip, width=180, height=320, fps=10)
    truth = motion_check.check_clip(clip, key)
    assert [f.t for f in truth.frames] == [0.0, 0.5, 1.0, 1.5, 1.9]
    assert truth.verdict == "pass", truth.reasons
    short = motion_check.check_clip(clip, key, until_s=1.0)
    assert short.frames[-1].t == pytest.approx(0.9)


def test_sample_times():
    assert motion_check.sample_times(2.0, 0.5) == [0.5, 1.0, 1.5, 2.0]
    assert motion_check.sample_times(1.9, 0.5) == [0.5, 1.0, 1.5, 1.9]


def _blobs():
    from tests.video.fakes import make_photo
    import tempfile
    from pathlib import Path

    return make_photo(Path(tempfile.mkdtemp()) / "o.jpg", (360, 640), seed=42)
