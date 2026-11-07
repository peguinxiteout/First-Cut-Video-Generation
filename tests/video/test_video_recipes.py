"""The recipes of configs/video/recipes.yaml: they load and validate, they only set brief fields and the prompt
box's words, their words read cleanly, and a card finds a photograph of its scene. Nothing here asks a model."""
from __future__ import annotations

import pytest
import yaml

from src.agents.video import prompt_brief, recipes
from src.core.config import load_yaml_config
from src.core.exceptions import ConfigError
from src.core.types import ImageCategory
from src.data.ingestion import discover_images
from src.data.schemas import ImageAnalysis, ImageAsset
from src.data.video_schemas import VIDEO_ASPECTS, VIDEO_DURATIONS, VIDEO_OBJECTIVES, VideoRecipe
from tests.video.helpers import make_video_property

EXPECTED = {
    "new_launch_teaser": (15, "launch", "confident"),
    "site_visit_invite": (15, "site_visit", "warm"),
    "amenity_tour": (30, "awareness", None),
    "green_living": (30, "awareness", None),
    "family_homes": (30, "awareness", None),
    "luxury_showcase": (30, "awareness", "refined"),
}


def test_the_six_recipes_load_in_order_and_validate():
    loaded = recipes.load()

    assert [r.id for r in loaded] == list(EXPECTED)
    for recipe in loaded:
        duration, objective, tone = EXPECTED[recipe.id]
        assert recipe.duration_s == duration and recipe.objective == objective
        assert tone is None or recipe.tone == tone
        assert recipe.duration_s in VIDEO_DURATIONS and recipe.aspect in VIDEO_ASPECTS
        assert recipe.objective in VIDEO_OBJECTIVES
        assert recipe.category in recipes.categories()
        assert "{property}" in recipe.prompt and recipe.description and recipe.name


def test_the_chips_are_the_categories_in_the_order_the_row_shows_them():
    assert list(recipes.categories().values()) == ["Launch", "Site visit", "Amenities", "Lifestyle", "Sustainability"]
    assert {r.category for r in recipes.load()} == set(recipes.categories())


def test_a_recipe_holds_only_the_fields_of_a_recipe_and_no_made_up_numbers():
    fields = set(VideoRecipe.model_fields)
    assert fields == {"id", "name", "category", "description", "duration_s", "aspect", "objective", "tone",
                      "music_mood", "cta", "prompt", "thumbnail"}
    raw = load_yaml_config("video/recipes.yaml")["recipes"]
    assert all(set(entry) <= fields for entry in raw)


def test_a_call_to_action_has_no_figure():
    for recipe in recipes.load():
        assert not any(ch.isdigit() for ch in recipe.cta), recipe.id


def test_the_recipes_set_brief_fields_only():
    values = recipes.brief_values(recipes.get("new_launch_teaser"))
    assert values == {"duration_s": 15, "aspect": "9:16", "objective": "launch", "tone": "confident",
                      "music_mood": "upbeat", "cta": "Register your interest today."}
    assert "cta" not in recipes.brief_values(recipes.get("amenity_tour"))               # optional
    assert set(recipes.brief_values(recipes.get("luxury_showcase"))) <= set(recipes.BRIEF_FIELDS)


def test_the_property_name_goes_in_the_words_and_a_missing_one_reads_plainly():
    recipe = recipes.get("site_visit_invite")

    assert "Brigade Citrine" in recipes.prompt_text(recipe, "Brigade Citrine")
    assert "{" not in recipes.prompt_text(recipe, "Brigade Citrine")
    assert "this property" in recipes.prompt_text(recipe)
    assert recipes.get("nope") is None


@pytest.mark.parametrize("recipe", recipes.load(), ids=lambda r: r.id)
def test_the_words_of_each_recipe_read_back_as_the_recipe_and_leave_nothing_out(recipe, tmp_path):
    folder = make_video_property(tmp_path / "data" / "properties")
    text = recipes.prompt_text(recipe, "Aurelia Heights")

    result = prompt_brief.read_prompt(text, folder, offline=True)

    assert result.brief.duration_s == recipe.duration_s
    assert result.brief.aspect in (None, recipe.aspect)
    assert result.brief.objective in (None, recipe.objective)              # the words never pull another way
    assert result.brief.tone in (None, recipe.tone)
    assert [n for n in result.notes if "left out" in n or "are not used" in n or "figure" in n] == []


def _use_config(tmp_path, monkeypatch, **changes):
    base = load_yaml_config("video/recipes.yaml")
    data = {**base, **changes}
    path = tmp_path / "recipes.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(recipes, "_config", lambda: yaml.safe_load(path.read_text(encoding="utf-8")))


def _first(**changes):
    return [{**load_yaml_config("video/recipes.yaml")["recipes"][0], **changes}]


@pytest.mark.parametrize("changes,said", [
    ({"duration_s": 20}, "duration_s"), ({"aspect": "4:3"}, "aspect"), ({"objective": "sell"}, "objective"),
    ({"tone": "loud"}, "tone"), ({"category": "nowhere"}, "category"), ({"thumbnail": "kitchen"}, "thumbnail"),
    ({"prompt": " "}, "prompt")])
def test_a_recipe_that_is_not_valid_is_named_in_words(tmp_path, monkeypatch, changes, said):
    _use_config(tmp_path, monkeypatch, recipes=_first(**changes))
    with pytest.raises(ConfigError, match=said):
        recipes.load()


def test_an_id_used_twice_is_refused(tmp_path, monkeypatch):
    entry = _first()[0]
    _use_config(tmp_path, monkeypatch, recipes=[entry, entry])
    with pytest.raises(ConfigError, match="twice"):
        recipes.load()


# -- the card's picture ---------------------------------------------------------------------------------------

def _images(*names):
    return [ImageAsset(image_id=f"IMG{n:03d}", filename=name, path=f"/x/{name}") for n, name in enumerate(names, 1)]


def _nothing_cached(monkeypatch):
    def classify(*args, **kwargs):
        raise RuntimeError("nothing is cached")

    monkeypatch.setattr(recipes.vision_cache, "classify", classify)


def test_a_card_gets_the_first_photograph_of_its_scene_from_the_file_names(monkeypatch):
    _nothing_cached(monkeypatch)
    pictures = recipes.scene_pictures(_images("aerial_view.webp", "tower_elevation_view.webp", "bedroom.webp",
                                              "pool_deck.webp", "gym_view.webp"))

    assert {scene: p.filename for scene, p in pictures.items()} == {
        "aerial": "aerial_view.webp", "exterior": "tower_elevation_view.webp", "interior": "bedroom.webp",
        "amenity": "pool_deck.webp"}
    assert recipes.for_scene(pictures, "amenity").filename == "pool_deck.webp"


def test_aerial_falls_to_an_exterior_and_a_scene_with_none_gives_nothing(monkeypatch):
    _nothing_cached(monkeypatch)
    pictures = recipes.scene_pictures(_images("tower_elevation_view.webp"))

    assert recipes.for_scene(pictures, "aerial").filename == "tower_elevation_view.webp"
    assert recipes.for_scene(pictures, "interior") is None
    assert recipes.for_scene({}, "exterior") is None


def test_the_reading_in_the_cache_decides_over_the_file_name(monkeypatch):
    from src.agents.video.vision_cache import Reading

    reading = Reading()
    reading.analyses["IMG001"] = ImageAnalysis(image_id="IMG001", category=ImageCategory.BEDROOM, quality=5,
                                               composition=5, hero_score=5)
    monkeypatch.setattr(recipes.vision_cache, "classify", lambda *args, **kwargs: reading)

    assert set(recipes.scene_pictures(_images("view_1.webp"))) == {"interior"}


def test_asking_for_thumbnails_never_calls_a_model(tmp_path, monkeypatch):
    folder = make_video_property(tmp_path / "data" / "properties")
    photos = [i for i in discover_images(folder) if i.role in ("", "photo")]
    asked = []
    monkeypatch.setattr("src.agents.video.vision_cache.classify_batch", lambda images: asked.append(images))

    recipes.scene_pictures(photos)

    assert asked == []
    with pytest.raises(RuntimeError):
        recipes._no_model(photos)
