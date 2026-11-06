"""Polish of the filled script and the scenes of its lines: frames used once while another fits, the property's own
wording and case, no area spoken, and the kind of picture each line sits over (rooms for the homes line, the aerial
picture for the scale line)."""
from __future__ import annotations

import math
import re
from collections import Counter

import pytest

from src.agents.video import brief as brief_agent
from src.agents.video import fact_ledger, photo_tags, vision_cache
from src.agents.video import script as script_agent
from src.agents.video import slots as video_slots
from src.agents.video import storyboard as sb
from src.core.config import load_yaml_config
from src.data.schemas import UnitType
from src.data.video_schemas import ScriptLine, Storyboard, VideoScript
from tests.video.helpers import Plan, make_video_property

UNITS = [UnitType(name="1 Bed + 1 Toilet", carpet_area="434.86 sq.ft"), UnitType(name="2 Bed + 2 Toilet", carpet_area="811.49 sq.ft")]
RICH = {"project_size": "4.3 Acres, 420 Units", "unit_types": UNITS}
MANY = "Swimming pool\nClubhouse\nGymnasium\nTennis court\nKids' play area\nLandscaped garden\nJogging track\nCCTV security"


def _writer(tmp_path, duration=30, updates=None, selling=None, **form):
    folder = make_video_property(tmp_path, **form)
    loaded = brief_agent.run(folder, {"duration_s": duration})
    prop = loaded.property_input.model_copy(update=updates or {})
    knowledge, _ = fact_ledger.run(prop, "", offline=True)
    if selling is not None:
        knowledge = knowledge.model_copy(update={"selling_points": selling})
    return script_agent._Writer(prop, loaded.brief, knowledge), prop, loaded.brief, knowledge


def _script(tmp_path, duration=30, updates=None, selling=None, **form) -> VideoScript:
    _, prop, brief, knowledge = _writer(tmp_path, duration, updates, selling, **form)
    return script_agent.run(prop, brief, knowledge, []).script


# -- variety ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tone", ["warm", "confident", "refined"])
def test_a_frame_is_used_once_in_a_video_while_another_that_fits_is_unused(tmp_path, tone):
    frames = load_yaml_config("video/script_templates.yaml")["tones"][tone]["zone"]
    items = ["Indoor Games", "Squash Court", "Swimming Pool", "Fitness Studio", "Yoga Deck", "Roof Garden", "Skate Park",
             "Cricket Net", "Open Lawn", "Dog Park"]
    w, *_ = _writer(tmp_path)
    for n in range(len(frames)):
        assert w.pick(frames, {**w.values, "item_list": items[n]})
        assert len(w.used) == n + 1                                    # each pick took a frame not yet used
    assert w.used == set(frames)
    assert w.pick(frames, {**w.values, "item_list": items[-1]})        # all used: one is used again, not none


def test_the_amenity_lines_of_a_video_do_not_share_a_frame_while_there_are_frames_left(tmp_path):
    frames = load_yaml_config("video/script_templates.yaml")["tones"]["warm"]["zone"]
    shapes = [re.compile("^" + re.escape(f).replace(re.escape("{item_list}"), "(.+)") + "$") for f in frames]
    script = _script(tmp_path, 60, amenities=MANY, usps="")
    zone = [l.text for l in script.lines if l.source.startswith("amenities:")]
    used = [next(i for i, shape in enumerate(shapes) if shape.match(text)) for text in zone]
    assert len(zone) >= 5
    assert max(Counter(used).values()) <= math.ceil(len(zone) / len(frames))
    if len(zone) <= len(frames):
        assert len(set(used)) == len(zone)


def test_the_hook_homes_and_call_to_action_have_two_or_three_frames_each():
    templates = load_yaml_config("video/script_templates.yaml")
    for tone, parts in templates["tones"].items():
        assert 2 <= len(parts["general"]["homes"]) <= 3, tone
        for objective in ("awareness", "site_visit", "launch", "enquiry"):
            assert 2 <= len(parts["hook"][objective]) <= 3, (tone, objective, "hook")
            assert 2 <= len(parts["cta"][objective]) <= 3, (tone, objective, "cta")


def test_the_same_inputs_give_the_same_script(tmp_path):
    first = _script(tmp_path / "a", 60, RICH, amenities=MANY)
    again = _script(tmp_path / "b", 60, RICH, amenities=MANY)
    assert [l.text for l in first.lines] == [l.text for l in again.lines]
    assert [l.scene for l in first.lines] == [l.scene for l in again.lines]


# -- the property's own wording --------------------------------------------------------------------------------------

def test_an_amenity_is_said_in_the_facts_own_words_and_case(tmp_path):
    script = _script(tmp_path, 60, usps="", amenities="Spa and Wellness\nIndoor Games\nBBQ and Dining\nSquash Court")
    said = " ".join(l.text for l in script.lines)
    for words in ("Spa and Wellness", "Indoor Games", "Squash Court"):
        assert words in said, words
    assert "spa and wellness" not in said                              # no lower-cased copy of a name


def test_a_name_the_facts_do_not_use_is_never_said(tmp_path):
    w, *_ = _writer(tmp_path, amenities="Fitness Studio", usps="")
    spa = script_agent.Feature("zone", "fitness", item_list="spa", chip="spa", amenity=True, units=("spa",))
    assert w.feature_line(spa) is None                                  # "spa" is a tag name here, not the property's word
    assert any("the property's facts do not call anything by that name" in n for n in w.notes)
    studio = script_agent.Feature("zone", "fitness", item_list="Fitness Studio", chip="Fitness Studio", amenity=True,
                                  units=("Fitness Studio",))
    assert w.feature_line(studio) is not None


# -- an area is never spoken ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Clubhouse of 25,000 sq ft", "1,200 sq.ft. decks", "Plots of 800 sqft", "A 3 sqm garden",
                                  "Spread over 4.3 acres", "2 hectares of park", "Carpet area 1017.62 sq. ft."])
def test_an_area_in_any_unit_is_found(text):
    assert video_slots.has_area(text)


def test_acres_are_allowed_only_for_the_scale_line():
    assert video_slots.has_area("420 homes across 4.3 acres") and not video_slots.has_area("420 homes across 4.3 acres", acres_ok=True)
    assert not video_slots.has_area("Sea views and a pool")


def test_a_selling_point_with_an_area_loses_the_clause_or_the_line(tmp_path):
    script = _script(tmp_path, 60, RICH, selling=["Sea views, 1,200 sq ft decks and a pool", "Clubhouse of 25,000 sq ft",
                                                  "Spread across 5.19 acres of parkland and lakes"])
    said = " ".join(l.text for l in script.lines)
    assert "sq ft" not in said and "25,000" not in said and "5.19" not in said and "1,200" not in said
    assert any(n.startswith("Selling point shortened: 'Sea views, 1,200 sq ft decks and a pool' to 'Sea views'") for n in script.notes)
    assert any(n.startswith("Selling point dropped: 'Clubhouse of 25,000 sq ft'") for n in script.notes)
    assert any(n.startswith("Selling point dropped: 'Spread across 5.19 acres") for n in script.notes)
    assert "420 homes across 4.3 acres." in said                       # the scale line keeps its acres


def test_a_clause_left_as_the_rest_of_a_sentence_is_dropped_not_said():
    assert video_slots.without_area("Spread across approximately 5.19 acres in the heart of Velachery, with 4 towers") == ""
    assert video_slots.without_area("Sea views, 3 sqm plots, gated entry") == "Sea views, gated entry"


def test_a_highlight_with_an_area_is_cut_or_dropped_with_a_note(tmp_path):
    w, *_ = _writer(tmp_path)
    assert w.without_area("Highlight", "Clubhouse of 25,000 sq ft") == ""
    assert w.without_area("Highlight", "Rooftop pool, 20,000 sq ft deck") == "Rooftop pool"
    assert any("Highlight dropped: 'Clubhouse of 25,000 sq ft' because it gives an area, which is not spoken." == n for n in w.notes)


# -- the scene of each line --------------------------------------------------------------------------------------------

def test_each_kind_of_line_asks_for_its_kind_of_picture(tmp_path):
    folder = make_video_property(tmp_path, amenities="Swimming pool\nGymnasium", usps="")
    plan = Plan(folder, {"duration_s": 60})
    plan.prop = plan.prop.model_copy(update=RICH)
    knowledge, _ = fact_ledger.run(plan.prop, "", offline=True)
    for asset in plan.assets:
        asset.tags = ["interior"] if asset.room else ["exterior"]
    script = script_agent.run(plan.prop, plan.brief, knowledge, plan.assets).script
    by = {l.source: l.scene for l in script.lines}
    assert script.lines[0].scene == "exterior" and script.lines[-1].scene == "any"
    assert by["unit types"] == "interior" and by["project size"] == "view" and by["possession"] == "any"
    assert all(l.scene == "amenity" for l in script.lines if l.source.startswith("amenities:"))


def test_the_homes_line_is_over_an_exterior_picture_when_no_room_is_shown(tmp_path):
    folder = make_video_property(tmp_path, usps="")
    plan = Plan(folder, {"duration_s": 45})
    plan.prop = plan.prop.model_copy(update=RICH)
    knowledge, _ = fact_ledger.run(plan.prop, "", offline=True)
    for asset in plan.assets:
        asset.tags, asset.room = ["exterior"], ""
    script = script_agent.run(plan.prop, plan.brief, knowledge, plan.assets).script
    homes = next(l for l in script.lines if l.source == "unit types")
    assert homes.scene == "exterior" and "inside" not in homes.text.lower()


# -- rooms: the stored category and the caption --------------------------------------------------------------------------

@pytest.mark.parametrize("category, caption, room", [
    ("living_room", "A living room with a brown sectional couch and botanical wall art.", "living"),
    ("living_room", "A dining room with an oval table, green chairs, and a balcony entrance", "dining"),   # the caption names it
    ("living_room", "A sunlit space with a sofa", "living"),                                              # the category does
    ("kitchen", "A kitchen with green cabinetry, stone countertops, and integrated oven", "kitchen"),
    ("bedroom", "A modern bedroom with sage green walls", "bedroom"),
    ("bathroom", "Marble vanity and a glass shower", "bathroom"),
    ("lifestyle", "A family reading in the study of a flat", "study"),
    ("other", "A balcony with two chairs", "balcony"),
    ("exterior", "Exterior balconies on an apartment building facing green outdoor areas.", ""),          # a building, not a room
    ("amenity", "A rooftop terrace with a dining table, outdoor kitchen counter, and pergola", ""),
    ("amenity", "A large double-height lobby lounge with marble flooring", ""),                           # not a home's room
    ("lifestyle", "A man and woman preparing food at an outdoor kitchen", ""),
    ("floorplan", "A plan of a living room and a kitchen", ""),
])
def test_a_room_is_read_from_the_stored_category_and_the_caption(category, caption, room):
    assert photo_tags.room(category, caption) == room


def test_the_assets_carry_their_room(tmp_path):
    plan = Plan(make_video_property(tmp_path))
    rooms = {a.filename: a.room for a in plan.assets}
    assert rooms["living_room.jpg"] == "living" and rooms["master_bedroom.jpg"] == "bedroom"
    assert rooms["kitchen.jpg"] == "kitchen" and rooms["exterior_facade.jpg"] == ""


def test_a_homes_dining_area_is_a_room_not_the_dining_amenity():
    assert photo_tags.from_caption("A cosy dining area in a flat with a large window") == []
    assert photo_tags.from_caption("An outdoor dining area with pergola seating") == ["dining", "pergola"]
    assert photo_tags.needs("Outdoor dining area") == {"dining"}                      # the property's own list is read as before
    assert photo_tags.room("living_room", "A cosy dining area in a flat") == "dining"


def test_the_rooms_are_not_part_of_the_vision_prompt_so_the_cache_stays_valid():
    prompt = vision_cache.prompt_text().lower()
    assert "dining area" not in prompt and "study room" not in prompt and "room_category" not in prompt
    assert vision_cache.prompt_version() == "2-b79939181c31"


# -- the pictures under the lines ----------------------------------------------------------------------------------------

ROOMS = {"exterior_facade.jpg": (["exterior"], ""), "swimming_pool.jpg": (["pool"], ""), "living_room.jpg": (["interior"], "living"),
         "master_bedroom.jpg": (["interior"], "bedroom"), "kitchen.jpg": (["interior"], "kitchen"),
         "clubhouse.jpg": (["lounge"], ""), "gym.jpg": (["gym"], ""), "garden_landscape.jpg": (["exterior"], "")}


def _plan(tmp_path, spec=ROOMS, duration=15) -> Plan:
    plan = Plan(make_video_property(tmp_path), {"duration_s": duration})
    for asset in plan.assets:
        asset.tags, asset.room = list(spec.get(asset.filename, ([], ""))[0]), spec.get(asset.filename, ([], ""))[1]
    return plan


def _board(plan, *lines) -> Storyboard:
    script = VideoScript(lines=list(lines), on_screen=["Aurelia Heights"], cta="Book a visit")
    return sb.run(plan.brief, script, plan.assets, plan.analyses, plan.image_plan)


def _by_id(plan):
    return {a.image_id: a for a in plan.assets}


HOOK = ScriptLine(text="Hi.", role="hook", scene="exterior")
CTA = ScriptLine(text="Book.", role="cta", scene="any")


def test_a_homes_line_plays_over_rooms_a_different_room_in_each_shot(tmp_path):
    plan = _plan(tmp_path)
    homes = ScriptLine(text="Step inside. 1, 2, 3 and 4 bedroom homes with plenty of room for a family.", role="proof",
                       topic="homes", scene="interior")
    price = ScriptLine(text="Prices from fifty lakh.", role="proof", topic="price", scene="any")
    board = _board(plan, HOOK, homes, price, CTA)
    mine = [_by_id(plan)[s.image_id] for s in board.shots if s.kind == "real" and 1 in s.line_ids]
    assert len(mine) == 2 and all(a.room for a in mine)
    assert len({a.room for a in mine}) == 2                                          # living, then another room
    assert mine[0].room == "living"


def test_a_line_that_wants_no_room_leaves_the_rooms_for_the_homes_line_still_to_come(tmp_path):
    plan = _plan(tmp_path)
    possession = ScriptLine(text="Possession in June 2029.", role="proof", topic="possession", scene="any")
    homes = ScriptLine(text="Step inside. 1, 2 and 3 bedroom homes.", role="proof", topic="homes", scene="interior")
    board = _board(plan, HOOK, possession, homes, CTA)
    by = _by_id(plan)
    first = next(s for s in board.shots if s.kind == "real" and 1 in s.line_ids)
    second = next(s for s in board.shots if s.kind == "real" and 2 in s.line_ids)
    assert not by[first.image_id].room and by[second.image_id].room


def test_a_homes_line_with_no_room_left_says_so(tmp_path):
    plan = _plan(tmp_path, {"exterior_facade.jpg": (["exterior"], ""), "swimming_pool.jpg": (["pool"], "")})
    homes = ScriptLine(text="Step inside. 1, 2 and 3 bedroom homes.", role="proof", topic="homes", scene="interior")
    board = _board(plan, HOOK, homes, CTA)
    assert any("not a room of a home: no unused photograph shows one" in n for n in board.notes)
    assert sb.validate_plan(board, plan.brief, plan.assets, plan.image_plan) == []


def test_a_scale_line_takes_the_aerial_picture_and_an_earlier_line_leaves_it(tmp_path):
    spec = {**ROOMS, "clubhouse.jpg": (["exterior"], ""), "gym.jpg": (["exterior"], "")}
    plan = _plan(tmp_path, spec)
    aerial = next(a for a in plan.assets if a.filename == "garden_landscape.jpg")
    aerial.topic = "aerial"
    selling = ScriptLine(text="India's first completely net zero community.", role="feature", scene="any")
    scale = ScriptLine(text="420 homes across 4.3 acres.", role="proof", topic="scale", scene="view")
    board = _board(plan, HOOK, selling, scale, CTA)
    assert next(s for s in board.shots if s.kind == "real" and 2 in s.line_ids).image_id == aerial.image_id
    assert aerial.image_id not in [s.image_id for s in board.shots if s.kind == "real" and 1 in s.line_ids]


def test_a_line_with_no_scene_takes_the_best_exterior_as_before(tmp_path):
    plan = _plan(tmp_path)
    by = _by_id(plan)
    candidates = [a.image_id for a in plan.assets if a.usable]
    line = ScriptLine(text="A line about nothing.", role="feature")
    choice = sb._by_tags([line], [], candidates, by)
    assert "exterior" in by[choice].tags and not by[choice].room
