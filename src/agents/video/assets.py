"""Node 3, asset audit: which of the uploaded pictures can carry a shot.

Reads each picture three ways and ranks them:

    what it is      the vision reading (a model when one is reachable, the
                    offline reading of the upload otherwise: its role, its
                    file name, its pixels)
    how big, how    its size, and its sharpness measured here: the variance
    sharp           of the Laplacian on a small greyscale copy
    what it looks   image_plan's visual groups, so two renders of the same
    like            elevation count once

Drawings (plans, maps, logos), pictures too small to fill a frame, and the
weaker twin of a pair are marked not usable, with the reason. When there are
fewer usable pictures than shots, that is said here, once, so the storyboard
and the person both know.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from PIL import Image, ImageOps

from src.agents.brochure import image_plan, topics, vision_offline
from src.agents.video import photo_tags, vision_cache
from src.agents.video._brochure import photo_score
from src.agents.video.offline import model_blocker
from src.core.config import load_yaml_config
from src.data.schemas import ImageAnalysis, ImageAsset
from src.data.video_schemas import RankedAsset

logger = logging.getLogger(__name__)

MEASURE_SIDE = 512            # the long side of the greyscale copy sharpness is measured on


def measure(path: str) -> tuple[int, int, float]:
    """(width, height, sharpness) of a picture; (0, 0, 0.0) if it cannot be read."""
    try:
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle)
            width, height = image.size
            grey = image.convert("L")
            grey.thumbnail((MEASURE_SIDE, MEASURE_SIDE))
            a = np.asarray(grey, dtype=np.float32)
    except Exception:  # noqa: BLE001 - an unreadable picture is reported, not fatal
        return 0, 0, 0.0
    if min(a.shape) < 3:
        return width, height, 0.0
    laplacian = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
    return width, height, float(laplacian.var())


def sharpness_score(variance: float) -> float:
    """0-10 on a log scale: a variance of 1,000 reads 10, 100 about 6.7, 10 about 3.5."""
    return round(min(10.0, math.log10(variance + 1) / 3 * 10), 1)


def read_pictures(images: list[ImageAsset], offline: bool) -> tuple[dict[str, ImageAnalysis], str]:
    """The vision reading of every picture, and how it was made."""
    blocker = model_blocker(offline, "vision")
    analyses: dict[str, ImageAnalysis] = {}
    mode = f"read from the uploads ({blocker})"
    if not blocker:
        try:
            reading = vision_cache.classify(images)
            analyses = reading.analyses
            mode = "read by a vision model" + (f" ({' '.join(reading.notes)})" if reading.notes else "")
        except Exception as exc:  # noqa: BLE001 - the uploads themselves tell enough to plan from
            logger.info("Vision unavailable (%s); reading the uploads offline", type(exc).__name__)
            mode = f"read from the uploads (vision unavailable: {type(exc).__name__})"
    missing = [i for i in images if i.image_id not in analyses]
    if missing:
        analyses.update(vision_offline.run(missing))
    return analyses, mode


def run(images: list[ImageAsset], offline: bool, photo_shots: int):
    """(analyses, ranked assets, image plan, notes). `photo_shots` is how
    many photographs the storyboard would like."""
    pacing = load_yaml_config("video/pacing.yaml")
    analyses, mode = read_pictures(images, offline)
    notes = [f"{len(images)} pictures {mode}."]

    measured = {i.image_id: measure(i.path) for i in images}
    for image_id, (_, _, variance) in measured.items():
        analysis = analyses[image_id]
        blended = round(min(10.0, (analysis.quality + sharpness_score(variance)) / 2), 1)
        analyses[image_id] = analysis.model_copy(update={"quality": blended})

    plan = image_plan.build({i.image_id: i for i in images}, analyses)
    ranked = []
    for image in images:
        width, height, variance = measured[image.image_id]
        analysis = analyses[image.image_id]
        category = getattr(analysis.category, "value", str(analysis.category))
        topic = image.subject.strip().lower() or topics.of_image(analysis, image.filename)   # a label wins
        tags, weak_tags = photo_tags.evidence(getattr(analysis, "tags", []), getattr(analysis, "caption", ""))
        asset = RankedAsset(
            image_id=image.image_id, filename=image.filename, path=image.path, width=width, height=height,
            sharpness=round(variance, 1), quality=analysis.quality, category=category, topic=topic,
            caption=getattr(analysis, "caption", ""),
            tags=tags, weak_tags=weak_tags, room=photo_tags.room(category, getattr(analysis, "caption", "")),
            cluster=plan.cluster(image.image_id))
        if width == 0:
            asset.usable, asset.note = False, "cannot be read"
        elif topic in topics.DRAWINGS or category in topics.DRAWINGS:
            asset.usable, asset.note = False, f"a {topic}, not a photograph"
        elif min(width, height) < pacing["min_photo_side_px"]:
            asset.usable, asset.note = False, f"too small ({width}x{height}) to fill a frame"
        ranked.append(asset)

    best_of: dict[int, RankedAsset] = {}
    for asset in sorted((a for a in ranked if a.usable), key=lambda a: -photo_score(a.image_id, analyses)):
        twin = best_of.setdefault(asset.cluster, asset) if asset.cluster >= 0 else None
        if twin is not None and twin is not asset:
            asset.usable, asset.duplicate_of = False, twin.image_id
            asset.note = f"looks like {twin.filename}"

    ranked.sort(key=lambda a: (not a.usable, -photo_score(a.image_id, analyses)))
    usable = [a for a in ranked if a.usable]
    skipped = [f"{a.filename} ({a.note})" for a in ranked if not a.usable]
    notes.append(f"{len(usable)} usable photographs" + (f"; set aside: {', '.join(skipped)}." if skipped else "."))
    if len(usable) < photo_shots:
        notes.append(f"Only {len(usable)} usable photograph(s) for {photo_shots} photo shots: the storyboard "
                     "will show a photograph again with a different crop, or use a card. "
                     "Add more photographs for a richer video.")
    return analyses, ranked, plan, notes
