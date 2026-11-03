from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from string import Template

from src.agents.base import with_retries
from src.core.config import load_yaml_config, settings
from src.core.exceptions import ModelOutputParseError
from src.core.types import ModelRole
from src.data.schemas import ImageAnalysis, ImageAsset
from src.models.gateway import call_with_fallback
from src.models.image_prep import prepare_images_for_vision_api

logger = logging.getLogger(__name__)

# Keep each batch request to a sane size — a handful of very large images in
# one call increases payload size and the chance the model garbles the
# per-image JSON array. 20 comfortably covers "one property's worth of
# photos" as a single call; a shoot larger than that gets chunked into a
# second (paced) batch call rather than falling back to one-call-per-photo.
MAX_IMAGES_PER_BATCH = 20


def _parse_batch_result(images: list[ImageAsset], raw_result: dict) -> dict[str, ImageAnalysis]:
    """Pure function (no I/O) so this is unit-testable without mocking the
    model gateway — see tests/agents/test_vision.py.

    Matches each result to its image by the explicit 1-based image_index
    every entry is required to include (see vision_classify_batch's
    prompt), NOT by array position. This is what makes a partial response
    safe to use: a real, observed failure mode is the model returning
    FEWER entries than images sent (e.g. 13 results for 16 images sent) —
    with an explicit index per entry, the images that DID get a result
    are still correctly identified even though some are missing, so the
    generation can proceed using whatever came back instead of hard-
    failing the entire batch over a handful of missing entries.

    Positional (zip-based) matching would be actively dangerous here: if
    the model drops image #5, every entry after it would silently shift
    by one position and get matched to the WRONG photo — a confidently
    wrong caption is worse than no caption at all. Falls back to
    positional matching only when the response has no usable indices AT
    ALL and the count happens to match exactly (unambiguous either way);
    otherwise a mismatched, index-less response is discarded for this
    batch rather than guessed at.
    """
    entries = raw_result.get("images")
    if not isinstance(entries, list) or not entries:
        raise ModelOutputParseError(f"Expected a JSON object with a non-empty 'images' array, got: {raw_result!r}")

    n = len(images)
    has_all_indices = all(isinstance(e, dict) and isinstance(e.get("image_index"), int) for e in entries)

    analyses: dict[str, ImageAnalysis] = {}

    if has_all_indices:
        seen_indices: set[int] = set()
        for entry in entries:
            index = entry["image_index"]
            if not (1 <= index <= n):
                logger.warning(
                    "Vision batch: ignoring a result with out-of-range image_index=%s (%d images were sent)",
                    index, n,
                )
                continue
            if index in seen_indices:
                logger.warning("Vision batch: duplicate image_index=%s in the response — keeping the first", index)
                continue
            seen_indices.add(index)
            image = images[index - 1]
            fields = {k: v for k, v in entry.items() if k != "image_index"}
            analyses[image.image_id] = ImageAnalysis(image_id=image.image_id, **fields)
    elif len(entries) == n:
        # No usable indices, but the count matches exactly — unambiguous,
        # so positional order (the original contract) is still safe here.
        for image, entry in zip(images, entries):
            fields = {k: v for k, v in entry.items() if k != "image_index"}
            analyses[image.image_id] = ImageAnalysis(image_id=image.image_id, **fields)
    else:
        # Neither a usable index nor a matching count — no safe way to
        # tell which image each entry belongs to. Guessing risks
        # confidently assigning a description to the wrong photo, which
        # is worse than this batch simply contributing no vision data;
        # every image just falls back to the offline reading of the upload
        # (vision_offline.py), which never invents a subject.
        logger.warning(
            "Vision batch: expected %d results, got %d, with no usable image_index in the response — "
            "discarding this batch's results rather than risk misaligned captions",
            n, len(entries),
        )
        return {}

    missing = n - len(analyses)
    if missing > 0:
        logger.warning(
            "Vision batch: %d of %d images got no result (the model returned fewer entries than were sent); "
            "those photos will use whatever text was already provided, with no Vision-assisted caption",
            missing, n,
        )

    return analyses


def _classify_batch(images: list[ImageAsset]) -> dict[str, ImageAnalysis]:
    prompt = Template(load_yaml_config("brochure/prompts.yaml")["vision_classify_batch"]).substitute(
        image_count=len(images)
    )
    original_paths = [image.path for image in images]

    # Send small, re-encoded copies to the model, never the originals —
    # see src/models/image_prep.py for why: a batch of full-resolution
    # photos can outright exceed a provider's payload limit, and even
    # short of that, sends far more bytes than classification needs.
    with tempfile.TemporaryDirectory(prefix="vision_batch_") as tmp_dir:
        resized_paths = prepare_images_for_vision_api(original_paths, Path(tmp_dir))
        raw_result = with_retries(
            lambda: call_with_fallback(ModelRole.VISION, "generate_json_with_images", prompt, resized_paths)
        )
    return _parse_batch_result(images, raw_result)


def run(images: list[ImageAsset]) -> dict[str, ImageAnalysis]:
    """Classifies + scores every image in as FEW LLM calls as possible
    (batched, not one-per-photo) — including the floor plan, which is
    expected to land in the 'floorplan' category via its visual content, you
    don't need to name the file specially.

    Batching matters a lot here: free-tier daily quotas can be as low as
    ~20 requests/day for a given model, and 12 individual per-photo calls
    would alone consume most of a single day's budget for one generation."""
    if not images:
        return {}

    analyses: dict[str, ImageAnalysis] = {}
    chunks = [images[i : i + MAX_IMAGES_PER_BATCH] for i in range(0, len(images), MAX_IMAGES_PER_BATCH)]
    for idx, chunk in enumerate(chunks):
        if idx > 0 and settings.vision_call_delay_seconds > 0:
            time.sleep(settings.vision_call_delay_seconds)
        analyses.update(_classify_batch(chunk))
    return analyses
