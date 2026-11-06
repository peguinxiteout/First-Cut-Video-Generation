"""The script follows the photographs: lines are chosen by what a picture shows (its tags), a chip is printed
only over a picture carrying its tag, and the hook shot lasts at least its minimum."""
from __future__ import annotations

import copy

import pytest

from src.agents.video import photo_tags
from src.agents.video import script as script_agent
from src.agents.video import storyboard as sb
from src.agents.video import voice
from src.data.video_schemas import ScriptLine, Shot, Storyboard, VideoScript
from tests.video.helpers import Plan, make_video_property

TAGS = {"exterior_facade.jpg": ["exterior"], "swimming_pool.jpg": ["pool"], "clubhouse.jpg": ["lounge", "interior"],
        "gym.jpg": ["gym"], "garden_landscape.jpg": ["lawn"], "living_room.jpg": ["interior"],
        "master_bedroom.jpg": ["interior"], "kitchen.jpg": ["interior"]}


def _plan(tmp_path, tags=TAGS, **overrides) -> Plan:
    """The plan half on a synthetic property, with the pictures tagged as a vision model would, and
    the script written again from them."""
    plan = Plan(make_video_property(tmp_path), overrides or None)
    tag(plan, tags)
    return plan


def tag(plan: Plan, tags) -> None:
    for asset in plan.assets:
        asset.tags = list(tags.get(asset.filename, []))
    plan.result = script_agent.run(plan.prop, plan.brief, plan.knowledge, plan.assets)
    plan.script = plan.result.script


def _by_id(plan):
    return {a.image_id: a for a in plan.assets}


def _board(plan, script=None):
    return sb.run(plan.brief, script or plan.script, plan.assets, plan.analyses, plan.image_plan)


# -- the script ---------------------------------------------------------------------------------------

def test_a_feature_line_about_an_amenity_has_a_photograph_that_shows_it(tmp_path):
    plan = _plan(tmp_path, highlights=["Rooftop infinity pool", "Gymnasium"])
    by_id = _by_id(plan)
    backed = [l for l in plan.script.lines if l.photo]

    assert backed, "the property has amenities with photographs, so some line is set against one"
    for line in backed:
        assert set(line.tags) <= set(by_id[line.photo].tags)
    assert len({l.photo for l in backed}) == len(backed)                      # each picture carries one line
    exterior = next(a for a in plan.assets if "exterior" in a.tags)
    assert exterior.image_id not in {l.photo for l in backed}                 # the hook's picture is not given away


def test_an_amenity_no_photograph_shows_is_left_out_with_a_note_not_set_over_another_picture(tmp_path):
    with_pool = _plan(tmp_path / "a", highlights=["Rooftop infinity pool"])
    assert any("pool" in l.text.lower() for l in with_pool.script.lines)

    no_pool = {**TAGS, "swimming_pool.jpg": ["lawn"]}                             # the pool picture shows no pool
    plan = _plan(tmp_path / "b", no_pool, highlights=["Rooftop infinity pool"])
    assert not any("pool" in l.text.lower() for l in plan.script.lines)
    assert any(n.startswith("Left out of the script: 'Rooftop infinity pool") and "No photograph shows it" in n
               for n in plan.script.notes)
    assert not any("pool" in c.lower() for c in plan.script.chips)


def test_a_general_line_needs_no_particular_picture(tmp_path):
    plan = _plan(tmp_path)
    general = [l for l in plan.script.lines if l.role == "feature" and not l.photo]
    assert all(l.tags == [] for l in general)
    assert [l.role for l in plan.script.lines][0] == "hook" and plan.script.lines[-1].role == "cta"


def test_without_tags_the_script_is_chosen_as_before(tmp_path):
    before = Plan(make_video_property(tmp_path), None)                       # offline: no vision, no tags
    assert not any(a.tags for a in before.assets)
    assert not any(l.photo or l.tags for l in before.script.lines)
    assert not [n for n in before.script.notes if n.startswith("Left out of the script")]
    tag(before, {})                                                           # tagging nothing changes nothing
    assert [l.text for l in before.script.lines] == [l.text for l in Plan(make_video_property(tmp_path / "again"), None).script.lines]


# -- the storyboard ---------------------------------------------------------------------------------------

def test_every_line_that_needs_a_picture_plays_over_one_that_shows_it(tmp_path):
    plan = _plan(tmp_path, highlights=["Rooftop infinity pool", "Gymnasium"])
    board = _board(plan)
    by_id = _by_id(plan)
    for shot in board.shots[:-1]:
        for index in shot.line_ids:
            line = plan.script.lines[index]
            if line.tags:
                assert set(line.tags) & set(by_id[shot.image_id].tags), (line.text, by_id[shot.image_id].filename)
    assert not any("which shows" in n for n in board.notes)
    assert "exterior" in by_id[board.shots[0].image_id].tags                  # the hook is an exterior


def _script(*lines: ScriptLine, chips=(), chip_lines=()) -> VideoScript:
    return VideoScript(lines=list(lines), on_screen=["Aurelia Heights"], chips=list(chips),
                       chip_lines=list(chip_lines), cta="Book a visit")


def _ids(plan):
    return {a.filename: a.image_id for a in plan.assets}


def test_a_chip_is_shown_over_a_picture_that_carries_its_tag(tmp_path):
    plan = _plan(tmp_path)
    ids = _ids(plan)
    script = _script(ScriptLine(text="Welcome.", role="hook", topic="exterior"),
                     ScriptLine(text="A pool on site.", role="feature", tags=["pool"], photo=ids["swimming_pool.jpg"]),
                     ScriptLine(text="Book a visit.", role="cta"), chips=["Swimming pool"], chip_lines=[1])
    board = _board(plan, script)
    shot = next(s for s in board.shots if "Swimming pool" in s.overlay_text)
    assert shot.image_id == ids["swimming_pool.jpg"] and 1 in shot.line_ids
    assert not any(n.startswith("Chip") for n in board.notes)


def test_a_chip_whose_tag_is_not_among_the_pictures_tags_is_dropped_with_the_reason(tmp_path):
    plan = _plan(tmp_path)
    ids = _ids(plan)
    script = _script(ScriptLine(text="Welcome.", role="hook", topic="exterior"),
                     ScriptLine(text="Barbecue evenings.", role="feature", tags=["gym"], photo=ids["gym.jpg"]),
                     ScriptLine(text="Book a visit.", role="cta"), chips=["BBQ"], chip_lines=[1])
    board = _board(plan, script)
    assert not any("BBQ" in s.overlay_text for s in board.shots)
    assert "Chip 'BBQ' dropped: gym.jpg (the picture under its line) shows gym, not bbq." in board.notes


def test_a_chip_that_names_nothing_a_picture_could_show_is_dropped(tmp_path):
    plan = _plan(tmp_path)
    ids = _ids(plan)
    script = _script(ScriptLine(text="Welcome.", role="hook", topic="exterior"),
                     ScriptLine(text="Good design.", role="feature", tags=["gym"], photo=ids["gym.jpg"]),
                     ScriptLine(text="Book a visit.", role="cta"), chips=["Award winning design"], chip_lines=[1])
    board = _board(plan, script)
    assert any(n.startswith("Chip 'Award winning design' dropped:") and "not what it names" in n for n in board.notes)


def test_untagged_pictures_keep_todays_chip_rule(tmp_path):
    plan = Plan(make_video_property(tmp_path), {"highlights": ["Rooftop infinity pool"]})     # offline
    for asset in plan.assets:
        asset.topic = "lifestyle"
    board = _board(plan)
    assert [t for s in board.shots[1:-1] for t in s.overlay_text] == []
    assert any(n.startswith("Chip 'Rooftop infinity pool' not shown: no photograph under its line") for n in board.notes)


def test_a_line_that_needs_a_picture_never_gets_an_unrelated_one_when_the_matching_pictures_are_used(tmp_path):
    plan = _plan(tmp_path)
    ids = _ids(plan)
    script = _script(ScriptLine(text="Welcome.", role="hook", topic="exterior"),
                     ScriptLine(text="Pool.", role="feature", tags=["pool"], photo=ids["swimming_pool.jpg"]),
                     ScriptLine(text="Pool again.", role="feature", tags=["pool"]),          # the only pool picture is taken
                     ScriptLine(text="Book a visit.", role="cta"))
    board = _board(plan, script)
    by_id = _by_id(plan)
    for shot in board.shots[:-1]:
        if shot.kind == "real" and 2 in shot.line_ids and 1 not in shot.line_ids:
            assert "pool" in by_id[shot.image_id].tags                         # shown again (marked reuse), never another view
    assert board.shots[-1].kind == "card"


# -- the hook ------------------------------------------------------------------------------------------------

CONFIG = {"end_card_s": 2.0, "min_hook_s": 2.5,
          "voice": {"lead_s": 0.3, "gap_s": 0.25, "tail_s": 0.3, "min_lead_s": 0.1, "min_gap_s": 0.1,
                    "min_end_card_s": 1.5, "pre_roll_s": 0.15, "fit_tolerance_s": 0.5, "min_other_shot_s": 1.0}}


def _board_of(*lines) -> Storyboard:
    shots = [Shot(index=i, seconds=3.0, line_ids=list(ids)) for i, ids in enumerate(lines[:-1])]
    shots.append(Shot(index=len(shots), kind="card", camera="still", seconds=2.0, line_ids=[len(lines) - 1]))
    return Storyboard(shots=shots)


def test_a_short_hook_line_still_gets_its_minimum_and_the_time_comes_from_the_other_shots():
    durations = [1.0, 2.5, 2.5, 2.0]                          # the hook line is short: its shot would be 1.7 s
    board = _board_of((0,), (1,), (2,), (3,))
    plain = voice.retime(durations, board, 10, {**CONFIG, "min_hook_s": 0.0})
    timing = voice.retime(durations, board, 10, CONFIG)

    assert plain.shot_seconds[0] < 2.5
    assert timing.shot_seconds[0] == pytest.approx(2.5, abs=1e-3)
    assert timing.total_s == plain.total_s and sum(timing.shot_seconds) == pytest.approx(timing.total_s, abs=0.01)
    assert abs(timing.total_s - 10) <= 0.5
    assert all(b <= a + 1e-6 for a, b in zip(plain.shot_seconds[1:-1], timing.shot_seconds[1:-1]))   # the others gave
    assert timing.shot_seconds[-1] == plain.shot_seconds[-1]                                          # the card is untouched
    assert min(timing.shot_seconds[1:-1]) >= 1.0 - 1e-6
    assert any("kept to 2.5 s" in n for n in timing.notes)


def test_a_hook_already_long_enough_is_left_alone():
    board = _board_of((0,), (1,), (2,), (3,))
    plain = voice.retime([2.5, 2.5, 2.5, 2.0], board, 15, {**CONFIG, "min_hook_s": 0.0})
    timing = voice.retime([2.5, 2.5, 2.5, 2.0], board, 15, CONFIG)
    assert plain.shot_seconds[0] > 2.5 and timing.shot_seconds == plain.shot_seconds
    assert not any("hook" in n for n in timing.notes)


def test_when_the_other_shots_have_no_time_to_give_the_hook_is_as_long_as_it_can_be_and_says_so():
    board = _board_of((0,), (1,), (2,), (3,))
    config = copy.deepcopy(CONFIG)
    config["voice"]["min_other_shot_s"] = 99.0                 # nothing may be taken
    timing = voice.retime([1.0, 2.5, 2.5, 2.0], board, 10, config)
    plain = voice.retime([1.0, 2.5, 2.5, 2.0], board, 10, {**config, "min_hook_s": 0.0})
    assert timing.shot_seconds == plain.shot_seconds
    assert any("under its minimum of 2.5 s" in n and "no time to give" in n for n in timing.notes)


def test_a_video_with_one_photograph_shot_is_not_changed_by_the_minimum():
    board = _board_of((0, 1), (2,))
    timing = voice.retime([1.0, 1.0, 1.0], board, 8, CONFIG)
    assert timing.shot_seconds == voice.retime([1.0, 1.0, 1.0], board, 8, {**CONFIG, "min_hook_s": 0.0}).shot_seconds


def test_a_shot_carries_its_pictures_caption_and_tags(tmp_path):
    plan = _plan(tmp_path)
    for asset in plan.assets:
        asset.caption = f"A picture of {asset.filename}."
    board = _board(plan)
    for shot in board.shots[:-1]:
        assert shot.tags == _by_id(plan)[shot.image_id].tags and shot.caption.startswith("A picture of ")
    assert board.shots[-1].tags == [] and board.shots[-1].caption == ""


# -- evidence: a tag needs its caption (photo_tags.evidence) --------------------------------------------------------

POOL_CAPTION = "A swimming pool with sun loungers and a deck"
TOWERS = "Two residential towers at dusk surrounded by green trees"       # a pool is in a corner of the frame


def read(plan: Plan, readings: dict) -> None:
    """Gives each picture the reading of a vision model, (its tags, its caption), through the evidence rule, and
    writes the script again."""
    for asset in plan.assets:
        tags, caption = readings.get(asset.filename, ([], ""))
        asset.tags, asset.weak_tags = photo_tags.evidence(tags, caption)
        asset.caption = caption
    plan.result = script_agent.run(plan.prop, plan.brief, plan.knowledge, plan.assets)
    plan.script = plan.result.script


AGREE = {"exterior_facade.jpg": (["exterior"], "A tower"), "swimming_pool.jpg": (["pool"], POOL_CAPTION),
         "gym.jpg": (["gym"], "A gym with treadmills"), "garden_landscape.jpg": (["lawn"], "A lawn with a garden")}


def test_a_photograph_tagged_and_captioned_with_the_amenity_is_used_for_its_line_and_chip(tmp_path):
    plan = Plan(make_video_property(tmp_path), {"highlights": ["Rooftop infinity pool"]})
    read(plan, AGREE)
    line = next(l for l in plan.script.lines if "pool" in l.text.lower())
    assert line.photo == _ids(plan)["swimming_pool.jpg"] and "pool" in line.tags
    assert any("pool" in c.lower() for c in plan.script.chips)
    assert not any("does not show it clearly" in n for n in plan.script.notes)


def test_a_tag_only_photograph_is_not_used_for_the_line_or_the_chip_and_the_note_says_so(tmp_path):
    plan = Plan(make_video_property(tmp_path), {"highlights": ["Rooftop infinity pool"]})
    read(plan, {**AGREE, "swimming_pool.jpg": (["pool"], TOWERS)})            # tagged pool, the caption does not say it
    ids = _ids(plan)

    assert not any("pool" in l.text.lower() for l in plan.script.lines)
    assert not any("pool" in c.lower() for c in plan.script.chips)
    assert not any(l.photo == ids["swimming_pool.jpg"] for l in plan.script.lines)
    assert ("Rooftop infinity pool left out: the only photograph tagged with it (swimming_pool.jpg) does not show it "
            "clearly.") in plan.script.notes
    assert not any(n.startswith("Left out of the script: 'Rooftop infinity pool") for n in plan.script.notes)


def test_the_tag_only_photograph_is_still_a_picture_for_a_scene_or_the_exterior(tmp_path):
    plan = Plan(make_video_property(tmp_path), {"highlights": ["Rooftop infinity pool"]})
    read(plan, {**AGREE, "swimming_pool.jpg": (["exterior", "pool"], TOWERS)})
    asset = next(a for a in plan.assets if a.filename == "swimming_pool.jpg")
    assert asset.tags == ["exterior"] and asset.weak_tags == ["pool"] and asset.usable      # still an exterior picture
    exteriors = [a.filename for a in plan.assets if "exterior" in a.tags]
    assert "swimming_pool.jpg" in exteriors


def test_when_several_photographs_are_tag_only_the_note_names_them_all(tmp_path):
    plan = Plan(make_video_property(tmp_path), {"highlights": ["Rooftop infinity pool"]})
    read(plan, {**AGREE, "swimming_pool.jpg": (["pool"], TOWERS), "clubhouse.jpg": (["pool", "interior"], "A hall")})
    note = next(n for n in plan.script.notes if n.startswith("Rooftop infinity pool left out:"))
    assert "the photographs tagged with it (" in note and "do not show it clearly" in note
    assert "swimming_pool.jpg" in note and "clubhouse.jpg" in note


def test_an_amenity_with_no_tag_at_all_keeps_the_old_note(tmp_path):
    plan = Plan(make_video_property(tmp_path), {"highlights": ["Rooftop infinity pool"]})
    read(plan, {**AGREE, "swimming_pool.jpg": (["lawn"], "A lawn with a garden")})
    assert any(n.startswith("Left out of the script: 'Rooftop infinity pool") and "No photograph shows it" in n
               for n in plan.script.notes)


def test_a_trimmed_line_says_which_dropped_unit_has_only_a_tag_only_photograph(tmp_path):
    plan = Plan(make_video_property(tmp_path), {"highlights": ["Gymnasium and swimming pool"]})
    read(plan, {**AGREE, "swimming_pool.jpg": (["pool"], TOWERS)})
    note = next(n for n in plan.script.notes if n.startswith("Trimmed 'Gymnasium and swimming pool'"))
    assert "to 'Gymnasium'" in note and "no photograph shows swimming pool" in note
    assert note.endswith("Tagged with it but not shown clearly: pool (swimming_pool.jpg).")


def test_a_chip_over_a_tag_only_photograph_is_dropped_and_says_why(tmp_path):
    plan = Plan(make_video_property(tmp_path), {"duration_s": 10})              # a hook shot, a pool shot, the card
    read(plan, {**AGREE, "swimming_pool.jpg": (["pool"], TOWERS)})
    ids = _ids(plan)
    script = _script(ScriptLine(text="Welcome.", role="hook", topic="exterior"),
                     ScriptLine(text="A pool on site.", role="feature", tags=["pool"], photo=ids["swimming_pool.jpg"]),
                     ScriptLine(text="Book a visit.", role="cta"), chips=["Swimming pool"], chip_lines=[1])
    board = _board(plan, script)

    assert not any("Swimming pool" in s.overlay_text for s in board.shots)
    assert ("Chip 'Swimming pool' dropped: swimming_pool.jpg (the picture under its line) is tagged with it but its "
            "caption does not say it, so it does not show it clearly.") in board.notes
