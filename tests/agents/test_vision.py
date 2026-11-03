from __future__ import annotations

import pytest

from src.agents.brochure.vision import _parse_batch_result
from src.core.exceptions import ModelOutputParseError
from src.data.schemas import ImageAsset


def _images(n: int) -> list[ImageAsset]:
    return [ImageAsset(image_id=f"IMG{i:03d}", filename=f"img{i}.jpg", path=f"/fake/img{i}.jpg") for i in range(n)]


def test_parse_batch_result_maps_entries_in_order_when_no_index_given():
    """Backward-compatible fallback: no image_index anywhere, but the count
    matches exactly, so positional order is unambiguous."""
    images = _images(2)
    raw_result = {
        "images": [
            {"category": "exterior", "quality": 8, "composition": 7, "hero_score": 9, "notes": "nice"},
            {"category": "floorplan", "quality": 6, "composition": 5, "hero_score": 1, "notes": "plan"},
        ]
    }
    parsed = _parse_batch_result(images, raw_result)
    assert parsed["IMG000"].category == "exterior"
    assert parsed["IMG000"].hero_score == 9
    assert parsed["IMG001"].category == "floorplan"


def test_parse_batch_result_uses_explicit_image_index():
    images = _images(2)
    raw_result = {
        "images": [
            {"image_index": 2, "category": "floorplan", "quality": 6, "composition": 5, "hero_score": 1, "notes": "plan"},
            {"image_index": 1, "category": "exterior", "quality": 8, "composition": 7, "hero_score": 9, "notes": "nice"},
        ]
    }
    parsed = _parse_batch_result(images, raw_result)
    assert parsed["IMG000"].category == "exterior"  # image_index 1 -> first image, despite arriving second
    assert parsed["IMG001"].category == "floorplan"


def test_parse_batch_result_handles_a_real_partial_response_without_raising():
    """Direct regression test for the actual reported bug: a batch of 16
    images where the model returned only 13 results. With explicit
    image_index on each entry, this must succeed with 13 correctly
    matched results instead of failing the whole generation."""
    images = _images(16)
    # Model skipped images 4, 9, and 15 (1-based) — arbitrary, not the last N.
    skipped = {4, 9, 15}
    raw_result = {
        "images": [
            {
                "image_index": i, "category": "living_room", "quality": 7, "composition": 7,
                "hero_score": 3, "notes": "x", "suggested_title": f"Room {i}",
            }
            for i in range(1, 17) if i not in skipped
        ]
    }
    parsed = _parse_batch_result(images, raw_result)

    assert len(parsed) == 13
    for i in skipped:
        assert images[i - 1].image_id not in parsed
    # Every result is matched to the CORRECT image, not shifted by the gaps.
    for i in range(1, 17):
        if i in skipped:
            continue
        assert parsed[images[i - 1].image_id].suggested_title == f"Room {i}"


def test_parse_batch_result_never_misaligns_when_it_cant_safely_match(caplog):
    """The dangerous case: count doesn't match AND there's no image_index
    to safely resolve it. Must return nothing rather than guess — a
    confidently wrong caption on the wrong photo is worse than none."""
    images = _images(3)
    raw_result = {
        "images": [
            {"category": "exterior", "quality": 8, "composition": 7, "hero_score": 9, "notes": "x"},
        ]
    }
    parsed = _parse_batch_result(images, raw_result)
    assert parsed == {}


def test_parse_batch_result_ignores_out_of_range_image_index():
    images = _images(2)
    raw_result = {
        "images": [
            {"image_index": 1, "category": "exterior", "quality": 8, "composition": 7, "hero_score": 9, "notes": "x"},
            {"image_index": 99, "category": "kitchen", "quality": 8, "composition": 7, "hero_score": 9, "notes": "y"},
        ]
    }
    parsed = _parse_batch_result(images, raw_result)
    assert len(parsed) == 1
    assert parsed["IMG000"].category == "exterior"


def test_parse_batch_result_keeps_the_first_of_a_duplicate_image_index():
    images = _images(1)
    raw_result = {
        "images": [
            {"image_index": 1, "category": "exterior", "quality": 8, "composition": 7, "hero_score": 9, "notes": "first"},
            {"image_index": 1, "category": "kitchen", "quality": 8, "composition": 7, "hero_score": 9, "notes": "second"},
        ]
    }
    parsed = _parse_batch_result(images, raw_result)
    assert len(parsed) == 1
    assert parsed["IMG000"].notes == "first"


def test_parse_batch_result_raises_when_images_key_missing():
    images = _images(1)
    with pytest.raises(ModelOutputParseError):
        _parse_batch_result(images, {"not_images": []})


def test_parse_batch_result_raises_when_images_key_is_not_a_list():
    images = _images(1)
    with pytest.raises(ModelOutputParseError):
        _parse_batch_result(images, {"images": "oops, a string not a list"})


def test_parse_batch_result_raises_when_images_list_is_empty():
    """A fully empty response is a broken call, not a partial one — still
    worth raising loudly rather than silently proceeding with zero data."""
    images = _images(2)
    with pytest.raises(ModelOutputParseError):
        _parse_batch_result(images, {"images": []})
