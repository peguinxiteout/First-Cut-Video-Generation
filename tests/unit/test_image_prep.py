from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.models.image_prep import (
    prepare_image_for_vision_api,
    prepare_images_for_vision_api,
    prepare_screenshot_for_vision_api,
)


def _make_image(path: Path, width: int, height: int, color=(120, 60, 200)) -> Path:
    Image.new("RGB", (width, height), color).save(path, format="JPEG", quality=95)
    return path


def test_large_photo_is_shrunk_and_re_encoded(tmp_path):
    """The actual bug this fixes: a batch of full-resolution photos
    exceeded a provider's 30MB payload limit outright."""
    source = _make_image(tmp_path / "big.jpg", 4000, 3000)
    original_size = source.stat().st_size

    result_path = prepare_image_for_vision_api(str(source), tmp_path / "out")

    assert result_path != str(source)  # a genuinely new, resized file
    with Image.open(result_path) as img:
        assert max(img.size) <= 1568
    assert Path(result_path).stat().st_size < original_size


def test_photo_aspect_ratio_is_preserved(tmp_path):
    source = _make_image(tmp_path / "wide.jpg", 4000, 2000)  # 2:1
    result_path = prepare_image_for_vision_api(str(source), tmp_path / "out")

    with Image.open(result_path) as img:
        ratio = img.width / img.height
        assert 1.9 <= ratio <= 2.1


def test_small_photo_is_not_upscaled(tmp_path):
    source = _make_image(tmp_path / "small.jpg", 200, 150)
    result_path = prepare_image_for_vision_api(str(source), tmp_path / "out")

    with Image.open(result_path) as img:
        assert img.size == (200, 150)


def test_batch_helper_resizes_every_image_in_order(tmp_path):
    paths = [
        str(_make_image(tmp_path / "a.jpg", 3000, 2000)),
        str(_make_image(tmp_path / "b.jpg", 500, 500)),
        str(_make_image(tmp_path / "c.jpg", 4500, 3000)),
    ]
    results = prepare_images_for_vision_api(paths, tmp_path / "out")

    assert len(results) == 3
    for r in results:
        with Image.open(r) as img:
            assert max(img.size) <= 1568


def test_resize_falls_back_to_original_path_on_a_corrupt_file(tmp_path):
    """A local resize failure must never block the whole generation —
    falls back to the original file rather than raising."""
    fake = tmp_path / "not_really_an_image.jpg"
    fake.write_bytes(b"this is not valid image data at all")

    result_path = prepare_image_for_vision_api(str(fake), tmp_path / "out")

    assert result_path == str(fake)


# ---------------------------------------------------------------------------
# Screenshot-specific: width-first capping, not longest-edge
# ---------------------------------------------------------------------------

def test_tall_multi_page_screenshot_is_capped_by_width_not_crushed_thin(tmp_path):
    """A 12-page brochure screenshot (Playwright's full_page=True stacks
    every page) is naturally very tall — capping its LONGEST edge (the
    height) would crush the width down to a useless sliver. Scaling must
    stay proportional (so it's still readable, not distorted), driven by
    whichever bound is tighter — here that's the height safety net, which
    pulls width down proportionally too, but nowhere near "crushed to a
    sliver."""
    source = _make_image(tmp_path / "screenshot.jpg", 1240, 15000)  # 12ish stacked pages

    result_path = prepare_screenshot_for_vision_api(str(source), tmp_path / "out")

    with Image.open(result_path) as img:
        assert img.height <= 12000  # capped by the height safety net
        # Proportionally scaled alongside height, not independently
        # crushed — still comfortably wide enough to read a page's
        # content, nowhere near a useless sliver.
        assert img.width >= 900
        # Aspect ratio preserved, not distorted.
        assert abs((img.width / img.height) - (1240 / 15000)) < 0.01


def test_wide_screenshot_is_capped_by_width(tmp_path):
    source = _make_image(tmp_path / "screenshot.jpg", 2400, 3000)
    result_path = prepare_screenshot_for_vision_api(str(source), tmp_path / "out")

    with Image.open(result_path) as img:
        assert img.width <= 1400


def test_screenshot_resize_falls_back_to_original_on_failure(tmp_path):
    fake = tmp_path / "broken.png"
    fake.write_bytes(b"not a real png")

    result_path = prepare_screenshot_for_vision_api(str(fake), tmp_path / "out")

    assert result_path == str(fake)


def test_a_large_batch_of_realistic_photos_stays_comfortably_under_30mb(tmp_path):
    """Direct regression test for the actual reported failure: OpenRouter
    rejected a batched vision request with 'Downloaded image content
    cannot exceed 30MB'. Simulates a realistic worst case — 20 large
    photos (MAX_IMAGES_PER_BATCH in vision.py) — and confirms the
    resized/re-encoded total stays well under that limit."""
    paths = []
    for i in range(20):
        # A high-resolution DSLR/phone-camera-sized photo with real noise
        # (not a flat color, which JPEG would compress to almost nothing —
        # random noise is a reasonably pessimistic stand-in for a detailed
        # real photo).
        import random
        img = Image.new("RGB", (4032, 3024))
        pixels = img.load()
        random.seed(i)
        for _ in range(50000):  # scatter enough noise to defeat trivial compression
            x, y = random.randint(0, 4031), random.randint(0, 3023)
            pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        path = tmp_path / f"photo_{i}.jpg"
        img.save(path, format="JPEG", quality=95)
        paths.append(str(path))

    results = prepare_images_for_vision_api(paths, tmp_path / "out")

    total_size = sum(Path(p).stat().st_size for p in results)
    assert total_size < 30 * 1024 * 1024, f"Batch payload is {total_size / 1024 / 1024:.1f}MB — still over the limit"
    # Should be dramatically smaller, not just barely under the wire.
    assert total_size < 10 * 1024 * 1024
