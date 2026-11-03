"""Type never lands on a photograph, and a photograph never loses its subject
to a crop — checked on the plan itself, for every template."""
from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from src.agents.brochure import catalog, focal, layouts, planner, qa, vision_offline
from src.agents.brochure.book import BookPage, Box, Placed
from src.data.ingestion import discover_images, load_property_input
from tests.brochure.helpers import make_property

CATEGORY = {"affordable": "Affordable", "mid": "Mid-Tier", "luxury": "Luxury"}
ORIENTATION = {"landscape": "Landscape", "portrait": "Portrait", "digital": "Digital 16:9"}


def _overlaps(a, b) -> bool:
    return min(a.x + a.w, b.x + b.w) - max(a.x, b.x) > 0.5 and min(a.y + a.h, b.y + b.h) - max(a.y, b.y) > 0.5


@pytest.mark.parametrize("template_id", list(catalog.TEMPLATES))
def test_no_text_box_overlaps_a_picture_and_no_deep_crop(tmp_path, template_id):
    spec = catalog.resolve(template_id)
    folder = make_property(tmp_path, category=CATEGORY[spec.category], orientation=ORIENTATION[spec.orientation])
    prop = load_property_input(folder)
    images = discover_images(folder)
    book = planner.plan(prop, images, vision_offline.run(images), spec)
    photos = [a for a in planner.inventory(images, vision_offline.run(images)) if a.kind == "photo"]
    for page in book.pages:
        for p in page.images:
            if page.kind == "cover":
                # the cover fills its sheet and is never reshaped; it takes the
                # photograph that fits the sheet best (within the cap when one does)
                best = min(layouts.crop_loss(a.aspect, p.w / p.h) for a in photos)
                assert p.crop <= max(layouts.CROP_CAP, best) + 0.08, (p.crop, best)
                continue
            if p.kind == "image":
                assert p.crop <= layouts.CROP_CAP + 1e-3, (page.kind, p.crop)
            for box in page.boxes.values():
                assert not _overlaps(p, box), (page.kind, p.slot, box.slot)


def _strip_page() -> BookPage:
    """An amenities page: a strip of photographs, a heading box and a text
    box side by side beneath it (the layout that once let a grown strip
    cover its heading)."""
    imgs = [Placed(slot=f"img{i}", image_id=f"I{i}", x=18 + i * 90, y=16, w=86, h=60, aspect=1.6, crop=0.1)
            for i in range(3)]
    return BookPage(kind="amenities", images=imgs,
                    boxes={"head": Box(slot="head", x=18, y=90, w=60, h=104),
                           "text": Box(slot="text", x=82, y=90, w=197, h=104)})


def test_enlarging_photographs_moves_every_box_beneath_them():
    page = _strip_page()
    assert qa.expand_pictures(page, {"head": 0.5, "text": 0.4})
    bottom = max(p.y + p.h for p in page.images)
    for box in page.boxes.values():
        assert box.y >= bottom - 0.01                       # nothing left under the grown strip
        assert not any(_overlaps(p, box) for p in page.images)
    assert all(p.crop <= layouts.CROP_CAP + 1e-3 for p in page.images)


def test_enlarging_is_refused_when_a_box_beneath_has_no_room():
    page = _strip_page()
    before = [(p.y, p.h) for p in page.images]
    assert not qa.expand_pictures(page, {"head": 0.97, "text": 0.4})   # the heading fills its box
    assert [(p.y, p.h) for p in page.images] == before


def test_an_over_deep_crop_reshapes_the_slot_instead():
    slot = layouts.Slot("img", catalog.Rect(0, 0, 210, 124), "image", (True, True, False, True))
    asset = planner.Asset(id="A", kind="photo", topic="arrival", aspect=0.71, px_w=2529, px_h=3576,
                          score=1, filename="gate.png")
    placed = planner._place(slot, asset, "image")
    assert placed.crop > layouts.CROP_CAP
    planner._cap_crop(placed, asset)
    assert placed.crop <= layouts.CROP_CAP + 1e-3
    assert placed.w < 210 and abs(placed.x + placed.w / 2 - 105) < 0.01     # narrowed about its centre
    assert placed.bleed[1] is False and placed.bleed[3] is False


def test_the_focal_point_follows_the_detail_not_the_sky(tmp_path):
    # a tall picture: flat sky over the top 60%, a detailed building below
    im = Image.new("RGB", (700, 1000), (150, 190, 230))
    draw = ImageDraw.Draw(im)
    for y in range(620, 980, 12):
        for x in range(20, 680, 24):
            draw.rectangle((x, y, x + 10, y + 6), fill=(90, 40, 30))
    path = tmp_path / "gate.png"
    im.save(path)
    position = focal.focal_point(str(path), 0.7, 1.7, "arrival")
    y = int(position.split()[1].rstrip("%"))
    assert y >= 70, position               # the window sits on the building, not the sky
