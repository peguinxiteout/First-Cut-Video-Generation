"""Vision for the asset audit: remembered per picture, and asked in parallel batches.

The brochure's vision agent (agents/brochure/vision.py) reads a property's pictures
in one or a few model calls. Two things make that expensive for a video, which
reads the same pictures again on every run:

    CACHE      a reading is kept in data/vision_cache/ under the picture's own
               bytes, the vision model and the prompt's version (the `version` in
               vision.yaml and a hash of the prompt text and tag names), so a second
               run of the same pictures asks nothing, and a changed picture, model
               or prompt is read afresh. A reading made before pictures were tagged
               is never reused.
    PARALLEL   the pictures still to read are split into batches of `batch_size`,
               and at most `concurrency` batches are in flight at once
               (configs/video/vision.yaml). The gateway's call policy travels with
               each batch (it lives in a context variable, which a thread does not
               inherit by itself).

A batch that fails costs only its own pictures: they get the offline reading of
their upload, and the pictures of the batches that worked are kept and cached.
If every batch fails the first error is raised, as before.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Callable, Optional

from src.agents.base import with_retries
from src.agents.brochure import vision
from src.agents.video import photo_tags
from src.core.config import load_yaml_config, settings
from src.core.types import ModelRole
from src.data.schemas import ImageAnalysis, ImageAsset
from src.data.video_schemas import PhotoAnalysis
from src.models.gateway import call_with_fallback
from src.models.image_prep import prepare_images_for_vision_api

logger = logging.getLogger(__name__)

CONFIG_FILE = "video/vision.yaml"
Runner = Callable[[list[ImageAsset]], dict[str, ImageAnalysis]]


@dataclass
class Reading:
    analyses: dict[str, ImageAnalysis] = field(default_factory=dict)
    cached: int = 0                                  # pictures answered from the cache
    asked: int = 0                                   # pictures the model was asked about
    batches: int = 0
    failed: dict[str, str] = field(default_factory=dict)         # image id -> why the model gave no reading
    notes: list[str] = field(default_factory=list)


def prompt_text() -> str:
    """The prompt a batch is asked: the brochure's classification prompt and the video's
    caption-and-tags addition, with $image_count still to fill."""
    config = load_yaml_config(CONFIG_FILE)
    extra = Template(config["tags_prompt"]).safe_substitute(
        max_tags=config.get("max_tags", 8), tag_names=", ".join(photo_tags.names()),
        scene_names=", ".join(photo_tags.scene_names()))
    return load_yaml_config("brochure/prompts.yaml")["vision_classify_batch"] + "\n" + extra


def prompt_version() -> str:
    """The prompt's version: vision.yaml's `version` and a hash of the prompt text, so a new
    prompt, new tag names or a raised version are a new cache."""
    digest = hashlib.sha256(prompt_text().encode("utf-8")).hexdigest()[:12]
    return f"{load_yaml_config(CONFIG_FILE).get('version', 1)}-{digest}"


def model_name() -> str:
    """The vision role's model, as the gateway would resolve it."""
    role = load_yaml_config("models.yaml")["roles"]["vision"]
    name = getattr(settings, role["model_env"].lower(), "") or role["model_env"]
    return f"{role['provider']}:{name}"


def cache_key(picture: bytes, model: str, version: str) -> str:
    payload = json.dumps({"image": hashlib.sha256(picture).hexdigest(), "model": model, "prompt": version},
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def cache_dir() -> Path:
    return settings.data_dir_resolved / "vision_cache"


def _load(path: Path, image_id: str) -> Optional[ImageAnalysis]:
    try:
        fields = json.loads(path.read_text(encoding="utf-8"))["analysis"]
        return PhotoAnalysis(image_id=image_id, **fields)
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Ignoring an unreadable vision cache entry: %s", path.name)
        return None


def _store(path: Path, analysis: ImageAnalysis, model: str, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = analysis.model_dump(mode="json", exclude={"image_id"})
    path.write_text(json.dumps({"model": model, "prompt_version": version, "analysis": fields}), encoding="utf-8")


def _tags_by_image(images: list[ImageAsset], raw_result: dict) -> dict[str, dict]:
    """{image id: {"caption", "tags"}} from the entries of a batch answer, matched by the
    entry's image_index exactly as the brochure parser matches them (or by position when
    none has an index and the count is right)."""
    entries = raw_result.get("images")
    if not isinstance(entries, list):
        return {}
    indexed = all(isinstance(e, dict) and isinstance(e.get("image_index"), int) for e in entries)
    if not indexed and len(entries) != len(images):
        return {}
    pairs = ([(images[e["image_index"] - 1], e) for e in entries if 1 <= e["image_index"] <= len(images)]
             if indexed else list(zip(images, entries)))
    limit = int(load_yaml_config(CONFIG_FILE).get("max_tags", 8))
    out: dict[str, dict] = {}
    for image, entry in pairs:
        if not isinstance(entry, dict):
            continue
        caption = entry.get("caption")
        out.setdefault(image.image_id, {
            "caption": " ".join(caption.split()) if isinstance(caption, str) else "",
            "tags": photo_tags.clean(entry.get("tags") if isinstance(entry.get("tags"), list) else [], limit)})
    return out


def classify_batch(images: list[ImageAsset]) -> dict[str, ImageAnalysis]:
    """One model call for a batch of pictures: the brochure's reading of each, with its
    caption and tags."""
    prompt = Template(prompt_text()).safe_substitute(image_count=len(images))
    originals = [image.path for image in images]
    with tempfile.TemporaryDirectory(prefix="vision_batch_") as tmp_dir:
        resized = prepare_images_for_vision_api(originals, Path(tmp_dir))
        raw = with_retries(lambda: call_with_fallback(ModelRole.VISION, "generate_json_with_images", prompt, resized))
    seen = _tags_by_image(images, raw)
    return {image_id: PhotoAnalysis(**analysis.model_dump(), **seen.get(image_id, {}))
            for image_id, analysis in vision._parse_batch_result(images, raw).items()}


def classify(images: list[ImageAsset], *, runner: Optional[Runner] = None, directory: Optional[Path] = None,
             model: Optional[str] = None, version: Optional[str] = None, concurrency: Optional[int] = None,
             batch_size: Optional[int] = None) -> Reading:
    """The vision reading of every picture it can get: from the cache, else from the model.
    Raises the first error only when nothing at all could be read."""
    config = load_yaml_config(CONFIG_FILE)
    runner = runner or classify_batch
    directory = directory if directory is not None else cache_dir()
    model, version = model or model_name(), version or prompt_version()
    concurrency = max(1, int(concurrency or config["concurrency"]))
    batch_size = max(1, int(batch_size or config["batch_size"]))
    reading = Reading()

    keys: dict[str, str] = {}                        # image id -> cache key
    for image in images:
        try:
            keys[image.image_id] = cache_key(Path(image.path).read_bytes(), model, version)
        except OSError:
            reading.failed[image.image_id] = "the picture could not be read"
    twins: dict[str, list[ImageAsset]] = {}           # one request per distinct picture
    for image in images:
        if image.image_id in keys:
            twins.setdefault(keys[image.image_id], []).append(image)

    todo: list[ImageAsset] = []
    for key, group in twins.items():
        hit = _load(directory / f"{key}.json", group[0].image_id) if (directory / f"{key}.json").exists() else None
        if hit is not None:
            for image in group:
                reading.analyses[image.image_id] = hit.model_copy(update={"image_id": image.image_id})
                reading.cached += 1
        else:
            todo.append(group[0])

    chunks = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    errors: list[Exception] = []
    if chunks:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(chunks))) as pool:
            futures = {pool.submit(contextvars.copy_context().run, runner, chunk): chunk for chunk in chunks}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    found = future.result()
                except Exception as exc:  # noqa: BLE001 - one batch failing must not lose the others
                    errors.append(exc)
                    for image in chunk:
                        reading.failed[image.image_id] = f"{type(exc).__name__}: {str(exc)[:120]}"
                    continue
                for image in chunk:
                    analysis = found.get(image.image_id)
                    if analysis is None:
                        reading.failed[image.image_id] = "the model gave no reading for it"
                        continue
                    if not isinstance(analysis, PhotoAnalysis):          # a reading without tags
                        analysis = PhotoAnalysis(**analysis.model_dump())
                    _store(directory / f"{keys[image.image_id]}.json", analysis, model, version)
                    for twin in twins[keys[image.image_id]]:
                        reading.analyses[twin.image_id] = analysis.model_copy(update={"image_id": twin.image_id})
        reading.batches = len(chunks)
        reading.asked = len(todo)
    if errors and not reading.analyses:
        raise errors[0]

    parts = []
    if reading.cached:
        parts.append(f"{reading.cached} from the cache")
    read = sum(1 for image in todo if image.image_id not in reading.failed)
    if read:
        parts.append(f"{read} read by the model in {reading.batches} batch(es), up to {concurrency} at a time")
    if parts:
        reading.notes.append("Vision: " + ", ".join(parts) + ".")
    if reading.failed:
        reading.notes.append(f"Vision gave no reading for {len(reading.failed)} picture(s) ("
                             f"{next(iter(reading.failed.values()))}); they were read from their uploads.")
    return reading
