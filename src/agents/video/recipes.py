"""Recipes for the Video tab: ready-made starts for real-estate videos (configs/video/recipes.yaml).

    load()                        -> the recipes, checked
    prompt_text(recipe, name)     -> the words for the prompt box, with the property's name in
    brief_values(recipe)          -> the brief fields the recipe sets (length and shape too)
    scene_pictures(images)        -> the property's photograph for each scene a card can show

A recipe sets brief fields and the prompt box's words and nothing else. It adds no fact: the script is
written from the property's own facts and photographs as for any other video. Where a recipe sits among the
other sources of an option is decided in src/ui/video_box.py (the toolbar, the form, the prompt, the recipe,
the property's video_input.json, the defaults).

The thumbnail of a card is a photograph of the chosen property: the vision reading is used when it is already
cached (this module never asks a model), else the file name's topic.
"""
from __future__ import annotations

import logging
from typing import Optional

from pydantic import ValidationError

from src.agents.brochure import topics
from src.agents.video import vision_cache
from src.core.config import load_yaml_config
from src.core.exceptions import ConfigError
from src.data.schemas import ImageAsset
from src.data.video_schemas import VideoRecipe

logger = logging.getLogger(__name__)

CONFIG_FILE = "video/recipes.yaml"
BRIEF_FIELDS = ("duration_s", "aspect", "objective", "tone", "music_mood", "cta")


def _config() -> dict:
    return load_yaml_config(CONFIG_FILE)


def load() -> list[VideoRecipe]:
    """Every recipe of the config, in its order. Raises ConfigError, in words, for an entry that is not valid, a
    category or scene the config does not define, or an id used twice."""
    cfg = _config()
    recipes: list[VideoRecipe] = []
    for entry in cfg.get("recipes") or []:
        try:
            recipes.append(VideoRecipe.model_validate(entry))
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise ConfigError(f"Recipe {entry.get('id', '?')!r} in {CONFIG_FILE} is not valid: {problems}.") from exc
    ids = [r.id for r in recipes]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{CONFIG_FILE} uses a recipe id twice.")
    tones = set(load_yaml_config("video/script_templates.yaml")["tones"])
    for recipe in recipes:
        if recipe.category not in categories():
            raise ConfigError(f"Recipe {recipe.id!r} has category {recipe.category!r}, which {CONFIG_FILE} does not define.")
        if recipe.thumbnail not in cfg["scenes"]:
            raise ConfigError(f"Recipe {recipe.id!r} has thumbnail {recipe.thumbnail!r}, which is not a scene of {CONFIG_FILE}.")
        if recipe.tone not in tones:
            raise ConfigError(f"Recipe {recipe.id!r} has tone {recipe.tone!r}; the tones are {', '.join(sorted(tones))}.")
    return recipes


def categories() -> dict[str, str]:
    """category id -> the chip's label, in the config's order."""
    return dict(_config()["categories"])


def get(recipe_id: str) -> Optional[VideoRecipe]:
    return next((r for r in load() if r.id == recipe_id), None)


def prompt_text(recipe: VideoRecipe, property_name: str = "") -> str:
    """The recipe's words for the prompt box, with the property's name where {property} is."""
    name = (property_name or "").strip() or _config()["no_property_name"]
    return recipe.prompt.replace("{property}", name)


def brief_values(recipe: VideoRecipe) -> dict:
    """The brief fields the recipe sets, by the names of VideoInput. A call to action is there only when the
    recipe has one."""
    return {field: getattr(recipe, field) for field in BRIEF_FIELDS if getattr(recipe, field)}


# ---------------------------------------------------------------------------
# The card's thumbnail
# ---------------------------------------------------------------------------

def _no_model(images):
    raise RuntimeError("no model is asked for a thumbnail")


def scene_of(topic: str) -> str:
    """The scene a picture topic belongs to ("" when it belongs to none)."""
    return next((scene for scene, names in _config()["scenes"].items() if topic in names), "")


def scene_pictures(images: list[ImageAsset]) -> dict[str, ImageAsset]:
    """The first photograph of each scene (file order) among `images`, which are the property's photographs.
    A scene with none is left out; callers use `for_scene` to apply the fallback."""
    try:
        analyses = vision_cache.classify(images, runner=_no_model).analyses
    except Exception:  # noqa: BLE001 - nothing cached is no reason to fail; the file names still say something
        analyses = {}
    found: dict[str, ImageAsset] = {}
    for image in images:
        scene = scene_of(topics.of_image(analyses.get(image.image_id), image.filename))
        if scene:
            found.setdefault(scene, image)
    return found


def for_scene(pictures: dict, scene: str):
    """The picture a card of `scene` shows: that scene's, else its fallback scene's, else None."""
    fallback = _config().get("scene_fallback") or {}
    return pictures.get(scene) or pictures.get(fallback.get(scene, ""))
