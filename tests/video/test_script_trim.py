"""A feature line says only the amenities a photograph shows: the units of a feature, the picture chosen for them, the
line rewritten to what it shows, and what is left out."""
from __future__ import annotations

import pytest

from src.agents.video import script as script_agent
from src.agents.video.script import Feature, best_fit
from src.data.video_schemas import RankedAsset
from tests.video.helpers import Plan, make_video_property


def _asset(name: str, *tags: str, quality=5.0) -> RankedAsset:
    return RankedAsset(image_id=name, filename=f"{name}.jpg", path=f"{name}.jpg", width=1600, height=1000, quality=quality,
                       tags=list(tags))


def _plan(tmp_path, tags: dict[str, list[str]], **overrides) -> Plan:
    plan = Plan(make_video_property(tmp_path), overrides or None)
    for asset in plan.assets:
        asset.tags = list(tags.get(asset.filename, []))
    plan.result = script_agent.run(plan.prop, plan.brief, plan.knowledge, plan.assets)
    plan.script = plan.result.script
    return plan


def _line(plan, word: str):
    return next((l for l in plan.script.lines if word in l.text.lower()), None)


# -- the units of a feature --------------------------------------------------------------------------------------------

def test_a_list_of_amenities_is_split_into_its_units():
    assert script_agent._units(["Multi-play Court and Outdoor Fitness Area"]) == ("Multi-play Court", "Outdoor Fitness Area")
    assert script_agent._units(["Pool, gym & spa"]) == ("Pool", "gym", "spa")
    assert script_agent._units(["Swimming pool", "Gymnasium"]) == ("Swimming pool", "Gymnasium")


def test_a_text_that_is_not_a_plain_list_stays_whole():
    assert script_agent._units(["Private jacuzzi for residents of the duplex residences"]) == (
        "Private jacuzzi for residents of the duplex residences",)
    assert script_agent._units(["A pool with a jacuzzi view"]) == ("A pool with a jacuzzi view",)    # two amenities, no way to split
    assert script_agent._units(["Two minutes from the metro and a quiet street"]) == (
        "Two minutes from the metro and a quiet street",)                                          # a part that names nothing: not a list


def test_the_picture_that_lets_a_feature_say_the_most_is_chosen_and_the_better_picture_wins_a_tie():
    feature = Feature("zone", "fitness", amenity=True, units=("Gymnasium", "Yoga"))
    both = _asset("both", "gym", "yoga", "lawn")
    one = _asset("one", "gym")
    fit = best_fit(feature, [one, both])
    assert fit.photo is both and fit.kept == ["Gymnasium", "Yoga"] and fit.dropped == []
    tie = best_fit(feature, [_asset("first", "gym"), _asset("second", "gym")])
    assert tie.photo.image_id == "first" and tie.kept == ["Gymnasium"] and tie.dropped == ["Yoga"]
    assert best_fit(feature, [_asset("none", "exterior")]) is None


def test_of_two_pictures_that_show_the_same_the_plainer_is_spent_and_the_richer_kept():
    feature = Feature("selling_point", "landscape", amenity=True, units=("Sixty percent open green space",))
    rich, plain = _asset("rich", "gym", "pool", "lawn"), _asset("plain", "lawn")
    assert best_fit(feature, [rich, plain]).photo is plain
    assert best_fit(feature, [plain, rich]).photo is plain


def test_a_unit_no_photograph_can_show_is_always_fine_to_say():
    feature = Feature("zone", "security", amenity=True, units=("Swimming pool", "24x7 CCTV security"))
    fit = best_fit(feature, [_asset("pool", "pool")])
    assert fit.kept == ["Swimming pool", "24x7 CCTV security"] and fit.dropped == []
    assert best_fit(feature, [_asset("lawn", "lawn")]) is None                      # but something visual has to be shown


def test_a_unit_naming_several_amenities_needs_a_picture_with_all_of_them():
    feature = Feature("highlight", "pool", amenity=True, units=("A pool with a jacuzzi view",))
    assert best_fit(feature, [_asset("pool", "pool")]) is None
    assert best_fit(feature, [_asset("both", "pool", "jacuzzi")]).kept == ["A pool with a jacuzzi view"]


# -- the line --------------------------------------------------------------------------------------------------------

def test_a_zone_line_is_trimmed_to_the_amenities_the_photograph_shows_and_the_trim_is_noted(tmp_path):
    plan = _plan(tmp_path, {"exterior_facade.jpg": ["exterior"], "gym.jpg": ["gym"]})
    line = _line(plan, "gymnasium")

    assert line is not None and "yoga" not in line.text.lower()
    assert line.tags == ["gym"] and line.photo == next(a.image_id for a in plan.assets if a.filename == "gym.jpg")
    assert "Trimmed 'gymnasium and yoga lawn' to 'gymnasium': no photograph shows yoga lawn." in plan.script.notes


def test_a_line_none_of_whose_amenities_has_a_photograph_is_dropped_and_noted(tmp_path):
    plan = _plan(tmp_path, {"exterior_facade.jpg": ["exterior"]})
    assert not any(l.photo for l in plan.script.lines)
    assert not any(word in " ".join(l.text.lower() for l in plan.script.lines) for word in ("gymnasium", "pool", "clubhouse"))
    left_out = [n for n in plan.script.notes if n.startswith("Left out of the script:")]
    assert any("gymnasium" in n.lower() for n in left_out) and any("swimming pool" in n.lower() for n in left_out)
    assert all(n.endswith("No photograph shows it.") for n in left_out)


def test_a_highlight_that_is_a_list_is_trimmed_part_by_part_and_its_chip_follows(tmp_path):
    plan = _plan(tmp_path, {"exterior_facade.jpg": ["exterior"], "swimming_pool.jpg": ["pool", "rooftop"]},
                 highlights=["Rooftop pool and jacuzzi"])
    line = _line(plan, "rooftop pool")
    assert line is not None and line.text == "Rooftop pool." and "jacuzzi" not in line.text.lower()
    assert "Trimmed 'Rooftop pool and jacuzzi' to 'Rooftop pool': no photograph shows jacuzzi." in plan.script.notes
    assert "Rooftop pool" in plan.script.chips and "Rooftop pool and jacuzzi" not in plan.script.chips


def test_a_highlight_that_cannot_be_split_is_kept_whole_or_left_out_whole(tmp_path):
    only_pool = _plan(tmp_path / "a", {"exterior_facade.jpg": ["exterior"], "swimming_pool.jpg": ["pool"]},
                      highlights=["A pool with a jacuzzi view"])
    assert not any("jacuzzi" in l.text.lower() for l in only_pool.script.lines)
    assert any(n.startswith("Left out of the script: 'A pool with a jacuzzi view'") for n in only_pool.script.notes)

    both = _plan(tmp_path / "b", {"exterior_facade.jpg": ["exterior"], "swimming_pool.jpg": ["pool", "jacuzzi"]},
                 highlights=["A pool with a jacuzzi view"])
    assert _line(both, "jacuzzi view") is not None
    assert not [n for n in both.script.notes if "jacuzzi view" in n]


def test_a_line_that_needs_no_picture_is_not_left_out_for_want_of_one(tmp_path):
    plan = _plan(tmp_path, {"exterior_facade.jpg": ["exterior"], "gym.jpg": ["gym"]}, highlights=["Round-the-clock CCTV security"])
    line = _line(plan, "cctv")
    assert line is not None and line.photo == "" and line.tags == []
    assert not [n for n in plan.script.notes if "cctv" in n.lower()]


def test_nothing_is_trimmed_when_the_photograph_shows_everything_the_line_says(tmp_path):
    plan = _plan(tmp_path, {"exterior_facade.jpg": ["exterior"], "gym.jpg": ["gym", "yoga", "lawn"],
                            "garden_landscape.jpg": ["lawn"]})            # the plain lawn picture goes to "green space"
    assert "yoga lawn" in _line(plan, "gymnasium").text.lower()
    assert not [n for n in plan.script.notes if n.startswith("Trimmed")]


def test_without_tags_no_line_is_trimmed_or_left_out(tmp_path):
    plan = Plan(make_video_property(tmp_path))                                         # offline: nothing is tagged
    assert not [n for n in plan.script.notes if n.startswith(("Trimmed", "Left out"))]
    assert all(l.photo == "" for l in plan.script.lines)
