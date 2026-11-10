"""What a photograph visibly shows: the tag vocabulary, and the vision batch that returns caption and tags."""
from __future__ import annotations

import pytest

from src.agents.video import photo_tags, vision_cache
from src.core.config import load_yaml_config, settings
from src.data.schemas import ImageAsset
from src.data.video_schemas import PhotoAnalysis
from tests.video.fakes import make_photo


# -- the vocabulary -------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Private jacuzzi for residents of the duplex residences.", {"jacuzzi"}),
    ("BBQ and dining, within the gates.", {"bbq", "dining"}),
    ("Swimming pool", {"pool"}),
    ("Rooftop infinity pool", {"pool", "rooftop"}),
    ("Kids' play area", {"play area"}),
    ("Clubhouse of 25,000 sq ft.", {"lounge"}),
    ("Multipurpose hall and party hall", {"hall"}),                        # a hall is a hall: not a lounge, not a pool
    ("Pool table and courtyard", {"lounge", "lawn"}),            # a pool table is not a pool; a courtyard is not a court
    ("Biophilic architecture designed in collaboration with SOG Design.", set()),
    ("Two minutes from the metro station.", set()),
])
def test_a_text_is_turned_into_the_amenities_it_names(text, expected):
    assert photo_tags.needs(text) == expected


def test_a_pictures_tags_are_merged_into_the_vocabulary_and_cleaned():
    assert photo_tags.clean(["Swimming Pool", "lap pool", "tower exterior", "trees", "outdoor seating", "bbq grill", "  ", 7]) \
        == ["pool", "exterior", "lawn", "outdoor seating", "bbq"]
    assert photo_tags.clean(None) == [] and len(photo_tags.clean([f"thing {i}" for i in range(20)])) == 8


def test_a_picture_shows_what_its_tags_name_and_nothing_it_only_resembles():
    lawn = ["lawn", "outdoor seating"]
    assert photo_tags.supports("BBQ and dining", ["bbq", "lawn"])
    assert not photo_tags.supports("BBQ", lawn)                         # a lawn is not a barbecue
    assert not photo_tags.supports("Private jacuzzi", ["lounge", "interior"])
    assert photo_tags.shows("Rooftop infinity pool", ["pool", "exterior"]) == ["pool"]
    assert photo_tags.supports("Outdoor seating by the lake", lawn)         # outside the vocabulary: a tag in the text
    assert not photo_tags.supports("Singapore-based design", ["exterior", "interior"])   # scenes never support a text
    assert not photo_tags.supports("Swimming pool", [])


def test_the_vision_prompt_offers_the_models_the_vocabulary():
    text = vision_cache.prompt_text()
    assert '"caption"' in text and '"tags"' in text and "CLEARLY" in text
    assert all(name in text for name in photo_tags.names()) and "exterior" in text
    assert "$" not in text.split("VIDEO FIELDS")[1]                  # every placeholder of the addition was filled


# -- the cache key follows the prompt version ------------------------------------------------------

def test_the_prompt_version_includes_the_version_in_the_config_and_the_prompt_text(monkeypatch):
    real = vision_cache.load_yaml_config
    base = vision_cache.prompt_version()
    assert base.startswith(f"{load_yaml_config('video/vision.yaml')['version']}-")

    monkeypatch.setattr(vision_cache, "load_yaml_config",
                        lambda name: {**real(name), "version": 99} if name == "video/vision.yaml" else real(name))
    bumped = vision_cache.prompt_version()
    assert bumped.startswith("99-") and bumped != base
    assert vision_cache.cache_key(b"picture", "m", bumped) != vision_cache.cache_key(b"picture", "m", base)


def test_a_reading_made_before_pictures_were_tagged_is_not_reused(tmp_path):
    """A cache entry written under the earlier prompt (the brochure prompt alone) has another key."""
    picture = make_photo(tmp_path / "p.jpg", (400, 300), seed=3).read_bytes()
    brochure_only = vision_cache.cache_key(picture, "gemini:m", "9c1f2a3b4c5d")
    assert vision_cache.cache_key(picture, "gemini:m", vision_cache.prompt_version()) != brochure_only


# -- the batch ---------------------------------------------------------------------------------------

def _images(tmp_path, n=3) -> list[ImageAsset]:
    out = []
    for i in range(n):
        path = make_photo(tmp_path / "photos" / f"p{i}.jpg", (400, 300), seed=i + 1)
        out.append(ImageAsset(image_id=f"IMG{i:03d}", filename=path.name, path=str(path), role="photo"))
    return out


def _entry(index, tags, caption="A view.", **extra):
    return {"image_index": index, "category": "amenity", "scene_tag": "pool_deck", "quality": 7.0, "composition": 7.0,
            "hero_score": 6.0, "caption": caption, "tags": tags, **extra}


def _ask(tmp_path, monkeypatch, entries, n=3):
    seen = {}

    def call(role, method, prompt, paths):
        seen["prompt"], seen["paths"] = prompt, list(paths)
        return {"images": entries}

    monkeypatch.setattr(vision_cache, "call_with_fallback", call)
    return vision_cache.classify_batch(_images(tmp_path, n)), seen


def test_a_batch_returns_the_caption_and_cleaned_tags_of_each_picture(tmp_path, monkeypatch):
    out, seen = _ask(tmp_path, monkeypatch, [
        _entry(1, ["Swimming Pool", "pool deck", "tower exterior"], caption="  A pool   beside a tower. "),
        _entry(2, ["lawn", "tree"]),
        _entry(3, []),
    ])
    assert "$image_count" not in seen["prompt"] and "3 real-estate" in seen["prompt"]
    assert "VIDEO FIELDS" in seen["prompt"] and len(seen["paths"]) == 3
    one, two, three = (out[f"IMG00{i}"] for i in range(3))
    assert isinstance(one, PhotoAnalysis) and (one.caption, one.tags) == ("A pool beside a tower.", ["pool", "exterior"])
    assert two.tags == ["lawn"] and three.tags == []
    assert one.category.value == "amenity" and one.quality == 7.0              # the brochure's reading is still there


def test_tags_are_matched_by_image_index_even_when_an_entry_is_missing(tmp_path, monkeypatch):
    out, _ = _ask(tmp_path, monkeypatch, [_entry(3, ["gym"]), _entry(1, ["pool"])])
    assert set(out) == {"IMG000", "IMG002"} and out["IMG000"].tags == ["pool"] and out["IMG002"].tags == ["gym"]


def test_an_answer_without_tags_gives_a_reading_with_none(tmp_path, monkeypatch):
    entries = [{k: v for k, v in _entry(i, []).items() if k not in ("caption", "tags")} for i in (1, 2, 3)]
    out, _ = _ask(tmp_path, monkeypatch, entries)
    assert all(a.tags == [] and a.caption == "" for a in out.values())


def test_a_malformed_tag_list_is_ignored_not_fatal(tmp_path, monkeypatch):
    out, _ = _ask(tmp_path, monkeypatch, [_entry(1, "pool"), _entry(2, [None, {"a": 1}, "gym"]), _entry(3, ["lawn"])])
    assert out["IMG000"].tags == [] and out["IMG001"].tags == ["gym"] and out["IMG002"].tags == ["lawn"]


def test_tags_survive_the_cache(tmp_path):
    images = _images(tmp_path, 2)

    def runner(chunk):
        return {i.image_id: PhotoAnalysis(image_id=i.image_id, category="amenity", quality=7, composition=7, hero_score=6,
                                          caption="A pool.", tags=["pool"]) for i in chunk}

    kwargs = dict(directory=tmp_path / "cache", model="gemini:m", version="v2")
    vision_cache.classify(images, runner=runner, **kwargs)
    again = vision_cache.classify(images, runner=lambda chunk: pytest.fail("asked again"), **kwargs)
    assert again.cached == 2 and again.analyses["IMG000"].tags == ["pool"] and again.analyses["IMG001"].caption == "A pool."


# -- captions, synonyms, and what no picture can show ------------------------------------------------------------------

@pytest.mark.parametrize("caption, expected", [
    ("Outdoor fitness equipment and children play structures set in a landscaped garden", ["gym", "play area", "lawn"]),
    ("An open-air gym area with treadmills and dumbbells.", ["gym"]),
    ("Two people holding squash rackets walking inside a squash court.", ["court"]),
    ("A rooftop terrace with a dining table, outdoor kitchen counter, and pergola.", ["dining", "rooftop", "pergola"]),
    ("A modern living room with a sectional sofa and decorative wall panels.", []),       # a sofa is not a lounge
    ("", []),
])
def test_a_caption_adds_the_amenities_it_speaks_of_and_never_a_view(caption, expected):
    assert photo_tags.from_caption(caption) == expected


BACKSIDE = "High-rise residential buildings illuminated at dusk surrounded by green trees"       # the pool is in a corner


def test_a_tag_stands_only_when_the_caption_says_it_too():
    assert photo_tags.evidence(["exterior", "pool"], "Two towers at dusk above a lit swimming pool") == (
        ["exterior", "pool"], [])
    assert photo_tags.evidence(["exterior", "pool"], BACKSIDE) == (["exterior"], ["pool"])       # weak: tag only


def test_the_caption_adds_no_tag_the_model_did_not_give():
    caption = "Outdoor fitness equipment and play structures set in a garden"
    assert photo_tags.from_caption(caption) == ["gym", "play area", "lawn"]
    assert photo_tags.evidence(["amenity", "play area", "lawn"], caption) == (["amenity", "play area", "lawn"], [])


def test_a_view_or_any_tag_that_is_not_an_amenity_needs_no_caption():
    assert photo_tags.evidence(["interior", "entrance", "exterior"], "") == (["interior", "entrance", "exterior"], [])
    assert photo_tags.evidence(None, None) == ([], []) and photo_tags.evidence(["a view"], "x") == (["a view"], [])


def test_without_a_caption_every_amenity_tag_is_weak():
    assert photo_tags.evidence(["pool", "gym", "exterior"], "") == (["exterior"], ["pool", "gym"])
    assert photo_tags.evidence(["pool", "gym", "exterior"], None) == (["exterior"], ["pool", "gym"])


def test_the_tags_are_cleaned_and_capped_before_the_caption_decides():
    shown, weak = photo_tags.evidence(["Swimming Pool", "pool", "lap pool", "gym"], "A lap pool", limit=12)
    assert shown == ["pool"] and weak == ["gym"]
    assert len(photo_tags.evidence([f"t{i}" for i in range(20)], "a gym", limit=12)[0]) == 12


@pytest.mark.parametrize("caption, expected", [
    (BACKSIDE, []),                                                      # trees are not a lawn
    ("A tree-lined promenade between two blocks", []),
    ("Greenery and open space around the towers", []),
    ("A courtyard with a bench", []),
    ("Children playing near the entrance", []),                          # children are not a play area
    ("A kids pool and a sun lounger", ["pool"]),                         # but a kids pool is a pool
    ("A wellness garden with a pergola", ["pergola", "lawn"]),           # wellness is not a spa
    ("A pond among the trees", ["water feature"]),                       # a water body is not a pool
    ("A dining room with an oval table, green chairs, and a balcony entrance", []),    # a home's dining room
    ("A rooftop terrace with an outdoor dining area", ["dining", "rooftop"]),
    ("A landscaped garden with a lawn and a park bench", ["lawn"]),
])
def test_a_caption_is_not_read_for_words_that_are_as_often_something_else(caption, expected):
    assert photo_tags.from_caption(caption) == expected


def test_the_words_a_caption_ignores_still_name_an_amenity_everywhere_else():
    assert photo_tags.needs("Landscaped garden with trees") == {"lawn"}
    assert photo_tags.needs("Kids play area") == {"play area"}
    assert photo_tags.needs("Wellness centre") == {"spa"}
    assert photo_tags.clean(["tree", "children"]) == ["lawn", "play area"]
    assert photo_tags.shows("Dining", ["dining"]) == ["dining"]


def test_the_caption_lists_are_not_part_of_the_prompt_the_vision_model_is_given():
    from src.agents.video import vision_cache

    prompt = vision_cache.prompt_text()
    assert "caption_ignore" not in prompt and "promenade" not in prompt and "dining room" not in prompt


@pytest.mark.parametrize("word, name", [
    ("treadmill", "gym"), ("dumbbells", "gym"), ("a swing set", "play area"), ("basketball hoop", "court"),
    ("yoga mat", "yoga"), ("outdoor dining", "dining"), ("jogging path", "track"), ("roof seating", "rooftop"),
])
def test_synonyms_for_what_a_camera_sees_are_words_of_the_names(word, name):
    assert name in photo_tags.needs(word)


def test_a_pet_park_is_not_just_a_park():
    assert photo_tags.needs("Pet Park") == set() and photo_tags.needs("a dog park") == set()
    assert photo_tags.needs("Park and Pet Park") == {"lawn"}                          # the plain park still reads as a lawn
    assert not photo_tags.supports("Pet Park", ["lawn", "play area"])               # a generic park does not show one
    assert photo_tags.supports("Pet Park", ["pet park"])                             # a picture tagged so does


def test_a_sofa_is_not_a_lounge_and_an_outdoor_kitchen_is_not_a_barbecue():
    assert photo_tags.needs("a sofa") == set() and photo_tags.needs("an outdoor kitchen counter") == set()


def test_synonyms_for_a_name_that_does_not_exist_are_ignored_and_do_not_break_the_vocabulary(monkeypatch, caplog):
    real = photo_tags.load_yaml_config
    photo_tags._vocabulary.cache_clear()
    monkeypatch.setattr(photo_tags, "load_yaml_config", lambda name: {**real(name), "synonyms": {"nope": ["x"], "gym": ["rowing machine"]}})
    try:
        assert photo_tags.needs("a rowing machine") == {"gym"}
        assert "synonyms for 'nope'" in caplog.text
    finally:
        monkeypatch.undo()
        photo_tags._vocabulary.cache_clear()
    assert photo_tags.needs("a rowing machine") == set()


@pytest.mark.parametrize("text, hidden", [
    ("24-hour security, CCTV monitoring, and automated boom barriers.", True),
    ("100% DG backup for both common areas and residential units.", True),
    ("Primary health center by Apollo Clinic and daycare by Eager Beavers.", True),
    ("EV-charging station, dedicated car wash area, and provision for stacked parking.", True),
    ("Swimming pool with CCTV cameras", False),                                    # it names an amenity a picture can show
    ("Multipurpose Hall", False), ("Biophilic architecture", False),
])
def test_a_text_naming_only_services_and_utilities_is_not_visual(text, hidden):
    assert photo_tags.non_visual(text) is hidden


def _vocabulary_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "vision_vocabulary.py"
    spec = importlib.util.spec_from_file_location("vision_vocabulary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_vocabulary_report_reads_every_property_json_and_says_what_is_not_yet_named(tmp_path):
    import json

    def write(name, amenities):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "property.json").write_text(json.dumps({"amenities": amenities}), encoding="utf-8")

    write("a", ["Swimming pool", "Skating rink", "24x7 CCTV security"])
    write("b", json.dumps(["Swimming pool", "Treadmill room"]))                  # the form stores the list as JSON text
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "property.json").write_text("{not json", encoding="utf-8")        # unreadable: skipped, not fatal
    module = _vocabulary_script()
    found = module.amenity_names(tmp_path)
    assert found["Swimming pool"] == ["a", "b"] and set(found) == {"Swimming pool", "Skating rink", "24x7 CCTV security",
                                                                  "Treadmill room"}
    named, hidden, rest = module.classify(sorted(found))
    assert named == ["Swimming pool", "Treadmill room"] and hidden == ["24x7 CCTV security"] and rest == ["Skating rink"]


def test_every_amenity_of_every_property_on_this_machine_is_named_or_not_visual():
    """The vocabulary was built from the union of data/properties/*/property.json (read only); a property
    added later with an amenity outside it shows up here and in scripts/vision_vocabulary.py."""
    from pathlib import Path

    folder = Path(__file__).resolve().parents[2] / "data" / "properties"         # the real one: tests use a temporary data folder
    if not folder.is_dir():
        pytest.skip("no data/properties folder on this machine")
    module = _vocabulary_script()
    _, _, rest = module.classify(sorted(module.amenity_names(folder)))
    assert rest == []
