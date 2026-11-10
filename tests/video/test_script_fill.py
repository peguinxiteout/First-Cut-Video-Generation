"""The script fills its length from the property's own facts: about a line for each photograph shot, general lines
about the homes, the scale and the place, selling points cut down (never reworded), no line under three words, no
topic twice in a row, and the shots and the closing card that go with it."""
from __future__ import annotations

import math

import pytest

from src.agents.video import brief as brief_agent
from src.agents.video import fact_ledger, photo_tags
from src.agents.video import script as script_agent
from src.agents.video import slots as video_slots
from src.agents.video import storyboard as sb
from src.core.config import load_yaml_config
from src.data.schemas import UnitType
from src.data.video_schemas import ScriptLine, VideoScript
from tests.video.helpers import Plan, make_video_property

UNITS = [UnitType(name="1 Bed + 1 Toilet", carpet_area="434.86 sq.ft"),
         UnitType(name="2 Bed + 2 Toilet", carpet_area="811.49 sq.ft"),
         UnitType(name="3 Bed + 2 Toilet", carpet_area="1017.62 sq.ft")]
HEADLINE = "India's first completely Net Zero Residential Community (Net Zero Carbon, Energy, Water, and Waste)."
PRIVACY = "Strict privacy design: Across all 420 units, zero homes face each other."
RICH = {"project_size": "4.3 Acres, 420 Units", "unit_types": UNITS}
HOMES_FRAMES = ("1, 2 and 3 bedroom homes.", "Homes of 1, 2 and 3 bedrooms.", "Choose from 1, 2 and 3 bedroom homes.")


def write(tmp_path, duration=30, updates=None, assets=None, tone="", selling=None, **form):
    """The script for a synthetic property, with `updates` laid over its fields. Without `assets` the pictures are
    unread (no tags), as in an offline run."""
    folder = make_video_property(tmp_path, **form)
    loaded = brief_agent.run(folder, {"duration_s": duration, **({"tone": tone} if tone else {})})
    prop = loaded.property_input.model_copy(update=updates or {})
    knowledge, _ = fact_ledger.run(prop, "", offline=True)
    if selling is not None:
        knowledge = knowledge.model_copy(update={"selling_points": selling})
    return script_agent.run(prop, loaded.brief, knowledge, assets or []).script, loaded.brief, prop


def texts(script: VideoScript) -> list[str]:
    return [line.text for line in script.lines]


# -- the claims: cut down by cutting only ------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    (HEADLINE, "India's first completely Net Zero Residential Community"),                    # the parenthetical goes
    (PRIVACY, "Across all 420 units, zero homes face each other"),                             # the label goes
    ("Gated community with 24x7 security.", "Gated community with 24x7 security"),
    ("Close to the metro (2 minutes): a short walk", "a short walk"),                           # a label of two words
    ("Quick access to the airport*", "Quick access to the airport"),
    ("Strategic Old Madras Road location offering excellent connectivity to Whitefield and the CBD", ""),   # over ten words
    ("100% of common area lighting is powered directly by renewable solar energy.", ""),        # eleven words
    ("Net Zero", ""),                                                                          # under three words
    ("A very long label that is more than six words long: Then the claim itself", ""),          # not a label: too long to cut
])
def test_a_selling_point_is_cut_down_by_cutting_never_by_rewording(text, expected):
    assert video_slots.claim(text) == expected


def test_a_claim_never_has_a_word_the_fact_did_not():
    for text in (HEADLINE, PRIVACY, "Quick access to the airport*"):
        said = video_slots.claim(text)
        bare = lambda t: set(t.lower().replace(",", "").replace(".", "").replace("*", "").split())    # noqa: E731
        assert said and bare(said) <= bare(text)


def test_the_headline_claim_is_a_line_of_its_own_words_with_its_source(tmp_path):
    script, _, _ = write(tmp_path, 30, RICH, selling=[HEADLINE, PRIVACY])
    by_source = {l.source: l for l in script.lines}
    headline = by_source["selling point: " + HEADLINE]
    assert headline.text == "India's first completely Net Zero Residential Community."
    assert by_source["selling point: " + PRIVACY].text == "Across all 420 units, zero homes face each other."
    for line in (headline, by_source["selling point: " + PRIVACY]):
        original = line.source.removeprefix("selling point: ").lower().replace(",", "")
        assert set(line.text.lower().replace(",", "").replace(".", "").split()) <= set(original.replace(".", "").split())


def test_a_claim_with_a_figure_that_is_not_a_fact_is_refused_by_the_guard(tmp_path):
    folder = make_video_property(tmp_path)
    loaded = brief_agent.run(folder, {"duration_s": 30})
    knowledge, _ = fact_ledger.run(loaded.property_input, "", offline=True)
    w = script_agent._Writer(loaded.property_input, loaded.brief, knowledge)
    w.values.update({"units": "999", "home_noun": "homes", "acres": "4.3"})
    assert w.general_line("scale", "scale", "project size") is None                           # 999 and 4.3 are not facts here
    assert any("is not a figure in the property's facts" in n for n in w.notes)


# -- the general lines --------------------------------------------------------------------------------------------

def test_each_general_line_is_written_from_its_own_facts(tmp_path):
    script, _, _ = write(tmp_path, 60, RICH)
    by_source = {l.source: l for l in script.lines if l.source in ("unit types", "project size", "location")}
    assert by_source["unit types"].text in HOMES_FRAMES
    assert by_source["project size"].text == "420 homes across 4.3 acres."
    assert by_source["location"].text == "At Baner, Pune."
    assert not any("sq ft" in l.text or "sq.ft" in l.text or "434" in l.text or "811" in l.text for l in script.lines
                   if l.source in ("unit types", "project size", "location"))


def test_a_general_line_is_skipped_when_its_slot_is_empty(tmp_path):
    script, _, _ = write(tmp_path, 60, {"project_size": "", "unit_types": []})
    assert not [l for l in script.lines if l.source in ("unit types", "project size")]
    one_type_without_beds = [*UNITS, UnitType(name="Penthouse", carpet_area="2000 sq.ft")]
    script, _, _ = write(tmp_path / "b", 60, {**RICH, "unit_types": one_type_without_beds})
    assert not [l for l in script.lines if l.source == "unit types"]                           # a list that leaves one out
    only_acres = write(tmp_path / "c", 60, {"project_size": "4.3 Acres", "unit_types": UNITS})[0]
    assert not [l for l in only_acres.lines if l.source == "project size"]                     # units are missing


def test_the_slots_are_the_properties_own_numbers():
    from src.data.schemas import PropertyInput, PropertyLocation

    p = PropertyInput(property_id="P", name="P", location=PropertyLocation(city="Pune"), unit_types=UNITS,
                      project_size="5.19 acres, 4 towers, 284 residences")
    assert video_slots.video_slots(p) == {"units": "284", "home_noun": "residences", "acres": "5.19", "towers": "4",
                                          "bed_list": "1, 2 and 3"}
    assert video_slots.video_slots(p.model_copy(update={"project_size": "1 Acre, 1 Tower", "unit_types": []})) == {
        "units": "", "home_noun": "", "acres": "", "towers": "", "bed_list": ""}
    assert video_slots.video_slots(p.model_copy(update={"project_size": "31 Storeys, 5 Blocks"}))["units"] == ""


def test_the_interior_connective_comes_before_the_homes_line_only_when_an_interior_photograph_is_left(tmp_path):
    plan = Plan(make_video_property(tmp_path), {"duration_s": 45})
    for asset in plan.assets:
        asset.tags, asset.room = ["exterior"], ""
    plain, _, _ = write(tmp_path / "a", 45, RICH, assets=plan.assets)
    line = next(l for l in plain.lines if l.source == "unit types")
    assert line.text in HOMES_FRAMES and line.scene == "exterior"                  # no room to show: no "inside"

    for asset in plan.assets[:2]:
        asset.tags, asset.room = ["interior"], "living"
    inside, _, _ = write(tmp_path / "b", 45, RICH, assets=plan.assets)
    line = next(l for l in inside.lines if l.source == "unit types")
    assert line.text in [f"{lead} {frame}" for lead in ("Step inside.", "Come inside.") for frame in HOMES_FRAMES]
    assert line.scene == "interior"


# -- no junk lines ------------------------------------------------------------------------------------------------

def test_a_proof_line_needs_three_words_so_a_lone_developer_name_is_not_a_line(tmp_path):
    folder = make_video_property(tmp_path, category="Luxury")
    loaded = brief_agent.run(folder, {"duration_s": 30})
    prop = loaded.property_input.model_copy(update={"developer": "Brigade"})
    knowledge, _ = fact_ledger.run(prop, "", offline=True)
    w = script_agent._Writer(prop, loaded.brief, knowledge)
    assert w.values["developer"] == "Brigade"
    assert w.pick(["{developer}."], min_words=3) == ""                                         # "Brigade." is one word
    assert w.pick(["From {developer}."], min_words=3) == ""
    assert w.pick(["Built by {developer}."], min_words=3) == "Built by Brigade."
    for tone in ("warm", "confident", "refined"):
        for seconds in (10, 15, 30, 45, 60):
            script, _, _ = write(tmp_path / f"{tone}{seconds}", seconds, {"developer": "Brigade"}, tone=tone)
            body = [l for l in script.lines if l.role not in ("hook", "cta")]
            assert all(l.words >= 3 for l in body), (tone, seconds, texts(script))
            assert "Brigade." not in texts(script)


# -- the line budget -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seconds", [10, 15, 30, 45, 60])
def test_the_script_fills_its_length_with_about_a_line_for_each_photograph_shot(tmp_path, seconds):
    script, brief, _ = write(tmp_path, seconds, RICH)
    target = brief.target_words
    shots = sb.photo_shot_count(brief)
    assert script.lines[0].role == "hook" and script.lines[-1].role == "cta"
    assert len(script.lines) - 1 <= shots + load_yaml_config("video/script_templates.yaml")["line_budget"]["extra_lines"]
    inside = math.floor(target * 0.9) <= script.word_count <= math.ceil(target * 1.1)
    assert inside or any("words against a target" in n for n in script.notes)       # outside the range, it says so
    if seconds <= 30:
        assert inside


def test_a_sixty_second_video_of_a_property_with_few_facts_says_it_is_short(tmp_path):
    script, brief, _ = write(tmp_path, 60, {"project_size": "", "unit_types": [], "usps": []}, usps="",
                             amenities="Parking", price="", possession="", configurations=[], nearby="")
    assert script.word_count < brief.target_words * 0.9
    assert any(f"words against a target of {brief.target_words}" in n for n in script.notes)


def test_shorter_videos_keep_working(tmp_path):
    for seconds in (10, 15):
        script, brief, _ = write(tmp_path / str(seconds), seconds, RICH)
        assert 3 <= len(script.lines) <= sb.photo_shot_count(brief) + 3
        assert script.word_count <= math.ceil(brief.target_words * 1.1)


def test_a_line_that_would_push_the_script_over_its_target_is_not_added(tmp_path):
    long = "A landscaped courtyard with seating and shade trees where neighbours meet every evening"
    script, brief, _ = write(tmp_path, 15, None, highlights=[long, long.replace("courtyard", "terrace")])
    assert script.word_count <= math.ceil(brief.target_words * 1.1)
    assert sum(long.split()[1] in t for t in texts(script)) <= 1


def test_every_line_whatever_its_source_passes_the_guard(tmp_path):
    from src.agents.video.guard import VideoGuard

    for seconds in (15, 30, 60):
        script, brief, prop = write(tmp_path / str(seconds), seconds, RICH, selling=[HEADLINE, PRIVACY])
        knowledge, _ = fact_ledger.run(prop, "", offline=True)
        knowledge = knowledge.model_copy(update={"selling_points": [HEADLINE, PRIVACY]})
        w = script_agent._Writer(prop, brief, knowledge)
        for line in script.lines:
            assert not w.guard.unknown_figure(line.text), line.text


# -- the order ----------------------------------------------------------------------------------------------------

def test_no_topic_follows_itself_and_the_hook_and_the_call_to_action_keep_their_places(tmp_path):
    script, _, _ = write(tmp_path, 60, RICH)
    body = script.lines[1:-1]
    assert script.lines[0].role == "hook" and script.lines[-1].role == "cta"
    for before, after in zip(body, body[1:]):
        assert not (before.topic and before.topic == after.topic), (before.text, after.text)


def test_the_spread_keeps_the_order_where_nothing_repeats_and_moves_only_what_must_move():
    def item(topic, text=""):
        return (ScriptLine(text=text or topic, role="feature", topic=topic), None)

    order = [item("a"), item("a", "a2"), item("b"), item("c")]
    assert [l.text for l, _ in script_agent._spread(order)] == ["a", "b", "a2", "c"]
    assert [l.text for l, _ in script_agent._spread([item("a"), item("b"), item("c")])] == ["a", "b", "c"]
    assert [l.text for l, _ in script_agent._spread([item("a"), item("a", "a2")])] == ["a", "a2"]    # nothing to go between
    assert [l.text for l, _ in script_agent._spread([item(""), item(""), item("")])] == ["", "", ""]    # no topic: no repeat


# -- the shots that go with it ---------------------------------------------------------------------------------------

def _board(plan: Plan, script: VideoScript):
    return sb.run(plan.brief, script, plan.assets, plan.analyses, plan.image_plan)


def test_each_line_gets_a_shot_and_the_hook_is_never_over_more_than_two(tmp_path):
    for seconds in (15, 30, 45):
        plan = Plan(make_video_property(tmp_path / str(seconds)), {"duration_s": seconds})
        board = _board(plan, plan.script)
        hook_shots = [s for s in board.shots if 0 in s.line_ids and s.kind == "real"]
        assert 1 <= len(hook_shots) <= 2
        body = {i for i, l in enumerate(plan.script.lines) if l.role != "cta"}
        assert body <= {i for s in board.shots for i in s.line_ids}                           # every line is on a shot


def test_spare_shots_go_to_the_lines_that_play_longest_and_the_hook_gets_at_most_two():
    lengths = [1.0, 4.0, 2.0, 9.0]
    shots = sb._a_shot_each([0, 1, 2, 3], lengths, 7)
    counts = [sum(1 for s in shots if s == [i]) for i in range(4)]
    assert sum(counts) == 7 and counts[3] > counts[1] > counts[2] >= 1 and counts[0] == 1
    long_hook = sb._a_shot_each([0, 1], [20.0, 1.0], 6)
    assert sum(1 for s in long_hook if s == [0]) == 2                                        # the hook stops at two


def test_with_more_lines_than_shots_the_shortest_neighbours_share_a_shot_but_never_two_that_need_pictures():
    script = VideoScript(lines=[ScriptLine(text="a", role="hook"), ScriptLine(text="b", role="feature", tags=["pool"]),
                                ScriptLine(text="c", role="feature"), ScriptLine(text="d", role="feature", tags=["gym"]),
                                ScriptLine(text="e", role="feature"), ScriptLine(text="f", role="cta")])
    merged = sb._merge_to_fit(script, [0, 1, 2, 3, 4], [2.0, 3.0, 1.0, 3.0, 1.5], 4)
    assert merged == [[0], [1, 2], [3], [4]]                                                  # b and c: 4.0 s, the shortest pair
    tagged = lambda g: sum(1 for i in g if script.lines[i].tags) <= 1                            # noqa: E731
    assert all(tagged(g) for g in merged)
    assert sb._merge_to_fit(script, [1, 3], [3.0, 3.0], 1) is None                           # both need a picture


def test_a_line_that_would_hold_a_still_past_the_cap_gets_a_second_shot(tmp_path):
    plan = Plan(make_video_property(tmp_path), {"duration_s": 30})
    long_line = "A line that a narrator needs about five seconds to say."                    # 11 words: 5.2 s and a gap
    short = [ScriptLine(text=f"Short line number {n} is here.", role="feature") for n in range(7)]
    script = VideoScript(lines=[ScriptLine(text="Hello there.", role="hook"), *short[:3],
                                ScriptLine(text=long_line, role="feature"), *short[3:],
                                ScriptLine(text="Book.", role="cta")])
    board = _board(plan, script)
    shots_of = lambda i: [s for s in board.shots if s.kind == "real" and s.line_ids == [i]]     # noqa: E731
    assert len(shots_of(4)) == 2                                                               # the long line, over two shots
    assert all(len(shots_of(i)) == 1 for i in (1, 2, 3, 5, 6, 7, 8))                           # the short ones keep one each
    assert max(s.seconds for s in board.shots if s.kind == "real") <= 4.5
