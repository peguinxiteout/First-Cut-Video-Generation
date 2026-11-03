"""What can be known about a picture without a vision model.

The Vision Agent (vision.py) is a model call. When it is unavailable — no
key, quota spent, a timeout — the brochure must still be planned sensibly,
so this reads what the pixels and the upload itself can tell us:

    the ROLE the person uploaded it as (a floor plan uploaded as a floor
             plan is one; a logo is a logo)
    a DRAWING from a photograph: line drawings and maps are mostly flat,
             pale and unsaturated, photographs are not
    the SUBJECT from the file name, when the name says it ("pool.jpg",
             "master-bedroom.png")
    the QUALITY from resolution and exposure

It never guesses a subject the name does not state, and never writes a
caption: a page with no known subject is given the general copy for the
project, not a description of a picture nobody has looked at.
"""
from __future__ import annotations

import logging

from src.agents.brochure import topics
from src.core.types import ImageCategory
from src.data.schemas import ImageAnalysis, ImageAsset

logger = logging.getLogger(__name__)

_ROLE_CATEGORY = {"floorplan": ImageCategory.FLOORPLAN, "masterplan": ImageCategory.FLOORPLAN,
                  "map": ImageCategory.MAP, "logo": ImageCategory.LOGO}
_TOPIC_CATEGORY = {
    "living": ImageCategory.LIVING_ROOM, "dining": ImageCategory.LIVING_ROOM,
    "kitchen": ImageCategory.KITCHEN, "bedroom": ImageCategory.BEDROOM,
    "bathroom": ImageCategory.BATHROOM, "balcony": ImageCategory.LIVING_ROOM,
    "exterior": ImageCategory.EXTERIOR, "aerial": ImageCategory.EXTERIOR,
    "night": ImageCategory.EXTERIOR, "arrival": ImageCategory.EXTERIOR,
    "lifestyle": ImageCategory.LIFESTYLE,
}


def _pixels(path: str):
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as im:
            im.draft("RGB", (256, 256))
            rgb = im.convert("RGB")
            w, h = im.size
            small = rgb.resize((96, 96))
            hsv = small.convert("HSV")
            sat = ImageStat.Stat(hsv.split()[1]).mean[0] / 255.0
            val = ImageStat.Stat(hsv.split()[2]).mean[0] / 255.0
            pale = sum(1 for p in small.convert("L").getdata() if p > 225) / (96 * 96)
            return w, h, sat, val, pale
    except Exception:  # noqa: BLE001
        return None


def analyse(image: ImageAsset) -> ImageAnalysis:
    stats = _pixels(image.path)
    role = (image.role or "").lower()
    name = image.filename or ""
    if role in _ROLE_CATEGORY:
        category = _ROLE_CATEGORY[role]
    elif stats and stats[4] > 0.55 and stats[2] < 0.18:
        category = ImageCategory.FLOORPLAN        # flat, pale, unsaturated: a drawing
    else:
        topic = topics.match(name)
        if topic in _TOPIC_CATEGORY:
            category = _TOPIC_CATEGORY[topic]
        elif topic:
            category = ImageCategory.AMENITY
        else:
            # No model and no name to go on: a neutral subject whose copy
            # suits any photograph of a home, rather than a guess.
            category = ImageCategory.LOGO if "logo" in name.lower() else ImageCategory.LIFESTYLE
    quality = 6.0
    if stats:
        w, h, sat, val, pale = stats
        megapixels = (w * h) / 1e6
        quality = 5.0 + min(3.0, megapixels) + (1.0 if 0.25 < val < 0.8 else 0.0)
    stem = name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
    return ImageAnalysis(
        image_id=image.image_id, category=category, quality=min(10.0, quality),
        composition=6.0, hero_score=6.5 if category == ImageCategory.EXTERIOR else 5.0,
        scene_tag=stem.lower().replace(" ", "_")[:60],
        suggested_title="", suggested_caption="", medium="", visible_text="",
    )


def run(images: list[ImageAsset]) -> dict[str, ImageAnalysis]:
    return {image.image_id: analyse(image) for image in images}
