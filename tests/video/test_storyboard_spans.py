"""A line that spans two shots uses two different photographs, or one longer shot; the same photograph is never in two
shots one after the other."""
from __future__ import annotations

import re

import pytest

from src.agents.video import storyboard as sb
from src.data.video_schemas import ScriptLine, Shot, Storyboard, VideoScript
from tests.video.helpers import Plan, make_video_property

TAGS = {"exterior_facade.jpg": ["exterior"], "swimming_pool.jpg": ["pool"], "gym.jpg": ["gym"],
        "garden_landscape.jpg": ["lawn"], "clubhouse.jpg": ["lounge"]}


def _tagged(tmp_path, tags=TAGS, **kwargs) -> Plan:
    plan = Plan(make_video_property(tmp_path, **kwargs))
    for asset in plan.assets:
        asset.tags = list(tags.get(asset.filename, []))
    return plan


def _ids(plan) -> dict[str, str]:
    return {a.filename: a.image_id for a in plan.assets}


def _board(plan, script) -> Storyboard:
    return sb.run(plan.brief, script, plan.assets, plan.analyses, plan.image_plan)


def _script(*lines: ScriptLine) -> VideoScript:
    return VideoScript(lines=list(lines), on_screen=["Aurelia Heights"], cta="Book a visit")


def _no_adjacent_repeat(board: Storyboard) -> bool:
    real = [s for s in board.shots if s.kind == "real"]
    return all(a.image_id != b.image_id for a, b in zip(real, real[1:]))


def test_a_line_over_two_shots_gets_two_different_photographs_both_showing_what_it_needs(tmp_path):
    plan = _tagged(tmp_path)
    ids = _ids(plan)
    line = ScriptLine(text="Pool and gym, on site, all day, all year round, for everyone who lives here.", role="feature",
                      tags=["pool", "gym"], photo=ids["swimming_pool.jpg"])
    board = _board(plan, _script(ScriptLine(text="Hi.", role="hook"), line, ScriptLine(text="Book.", role="cta")))

    carrying = [s for s in board.shots if 1 in s.line_ids]
    assert len(carrying) == 2                                                        # the third shot's time went to the second
    assert len({s.image_id for s in carrying}) == len(carrying)                     # different photographs
    assert all(set(s.tags) & {"pool", "gym"} for s in carrying)                      # each showing what the line needs
    assert _no_adjacent_repeat(board) and sb.validate_plan(board, plan.brief, plan.assets, plan.image_plan) == []


def test_when_no_other_photograph_shows_it_the_two_shots_become_one_longer_shot(tmp_path):
    plan = _tagged(tmp_path)
    ids = _ids(plan)
    line = ScriptLine(text="The pool, on site, open all day, all year round, for everyone who lives here.", role="feature",
                      tags=["pool"], photo=ids["swimming_pool.jpg"])                  # only one picture shows a pool
    board = _board(plan, _script(ScriptLine(text="Hi.", role="hook"), line, ScriptLine(text="Book.", role="cta")))

    carrying = [s for s in board.shots if 1 in s.line_ids]
    assert len(carrying) == 1 and carrying[0].image_id == ids["swimming_pool.jpg"]
    assert carrying[0].seconds > 4.0                                                 # it took the time of the shot it replaced
    assert len(board.shots) < plan.brief.n_shots and abs(board.total_seconds - 15.0) < 0.1
    assert any("joined to the shot before" in n and "pool" in n for n in board.notes)
    assert _no_adjacent_repeat(board) and sb.validate_plan(board, plan.brief, plan.assets, plan.image_plan) == []


def test_a_joined_shot_is_named_by_its_line_and_says_how_long_the_hold_now_is(tmp_path):
    plan = _tagged(tmp_path)
    ids = _ids(plan)
    line = ScriptLine(text="The pool, on site, open all day, all year round, for everyone who lives here.", role="feature",
                      tags=["pool"], photo=ids["swimming_pool.jpg"])
    board = _board(plan, _script(ScriptLine(text="Hi.", role="hook"), line, ScriptLine(text="Book.", role="cta")))

    notes = [n for n in board.notes if "is joined to the shot before" in n]
    assert notes and any(n.startswith("The shot for 'The pool, on site, open all day, all year round, for everyone "
                                      "who lives here.' is joined to the shot before, which is now held ") for n in notes)
    assert any("no unused photograph shows pool or fits as a general picture (swimming_pool.jpg stays on screen)" in n
               for n in notes)
    assert not any(re.search(r"\bShot \d", n) for n in notes)     # the shot it would have been is never made: no number to match


def test_a_merged_shot_carries_the_lines_of_both_and_keeps_the_chip_of_the_first(tmp_path):
    plan = _tagged(tmp_path)
    ids = _ids(plan)
    line = ScriptLine(text="The pool, on site, open all day, all year round, for everyone who lives here.", role="feature",
                      tags=["pool"], photo=ids["swimming_pool.jpg"])
    script = _script(ScriptLine(text="Hi.", role="hook"), line, ScriptLine(text="Book.", role="cta"))
    script.chips, script.chip_lines = ["Swimming pool"], [1]
    board = _board(plan, script)
    shot = next(s for s in board.shots if 1 in s.line_ids)
    assert shot.overlay_text == ["Swimming pool"] and sum(1 for s in board.shots if "Swimming pool" in s.overlay_text) == 1


def test_the_hook_is_the_plainest_exterior_so_an_exterior_that_also_shows_a_pool_is_kept_for_the_pool(tmp_path):
    tags = {**TAGS, "exterior_facade.jpg": ["exterior", "pool", "lawn"], "living_room.jpg": ["exterior"]}
    plan = _tagged(tmp_path, tags)
    usable = [a for a in plan.assets if a.usable]
    hook = sb.hook_asset(usable, "", [])
    assert hook.filename == "living_room.jpg"                                      # one tag, not three
    only = _tagged(tmp_path / "again", {**TAGS, "exterior_facade.jpg": ["exterior", "pool"]})
    assert sb.hook_asset([a for a in only.assets if a.usable], "", []).filename == "exterior_facade.jpg"   # the only exterior is the hook
    assert sb.hook_asset([a for a in only.assets if a.usable], "swimming_pool", []).filename == "swimming_pool.jpg"   # a named hero wins


def test_a_line_with_no_needs_over_two_shots_takes_a_different_photograph_for_each(tmp_path):
    plan = _tagged(tmp_path)
    general = ScriptLine(text="Designed for people who love to live well, close to everything the city has to offer.",
                         role="feature")
    board = _board(plan, _script(ScriptLine(text="Hi.", role="hook"), general, ScriptLine(text="Book.", role="cta")))
    assert _no_adjacent_repeat(board)
    assert len({s.image_id for s in board.shots if s.kind == "real"}) == len([s for s in board.shots if s.kind == "real"])


@pytest.mark.parametrize("photos", [2, 3, 4])
def test_with_few_photographs_a_photograph_comes_back_later_never_straight_after(tmp_path, photos):
    plan = Plan(make_video_property(tmp_path, photos=photos))                       # offline: no tags
    for shots_wanted in (plan.script,):
        board = _board(plan, shots_wanted)
        assert _no_adjacent_repeat(board)
        assert sb.validate_plan(board, plan.brief, plan.assets, plan.image_plan) == []


def test_the_check_every_plan_has_to_pass_refuses_the_same_photograph_in_two_shots_in_a_row(tmp_path):
    plan = _tagged(tmp_path)
    first = next(a for a in plan.assets if a.usable)
    shots = [Shot(index=0, seconds=4.0, image_id=first.image_id, source_path=first.path),
             Shot(index=1, seconds=4.0, image_id=first.image_id, source_path=first.path, reuse=True),
             Shot(index=2, seconds=5.0, image_id=first.image_id, source_path=first.path, reuse=True),
             Shot(index=3, kind="card", camera="still", seconds=2.0)]
    problems = sb.validate_plan(Storyboard(shots=shots), plan.brief, plan.assets, plan.image_plan)
    assert any(p.startswith("Shots 1 and 2 show the same photograph") and first.filename in p for p in problems)
    assert any(p.startswith("Shots 2 and 3 show the same photograph") for p in problems)


# -- a continuation takes another photograph instead of stretching the shot before ----------------------------------

GENERAL = {"exterior_facade.jpg": ["exterior"], "swimming_pool.jpg": ["exterior", "pool"],
           "garden_landscape.jpg": ["exterior"], "clubhouse.jpg": ["exterior", "entrance"], "gym.jpg": ["gym"],
           "living_room.jpg": ["interior"], "master_bedroom.jpg": ["interior"], "kitchen.jpg": ["interior"]}
LONG_POOL = ("The pool, on site, open all day, all year round, for everyone who lives here and for the guests "
             "who come to stay.")


def _general_plan(tmp_path, tags=GENERAL, **kwargs) -> Plan:
    plan = _tagged(tmp_path, tags, **kwargs)
    plan.brief = plan.brief.model_copy(update={"hero_image": "exterior_facade.jpg"})       # the hook is not left to a tie
    return plan


def _pool_script(ids, photo="swimming_pool.jpg", *more: ScriptLine) -> VideoScript:
    return _script(ScriptLine(text="Hi.", role="hook"),
                   ScriptLine(text=LONG_POOL, role="feature", tags=["pool"], photo=ids[photo]), *more,
                   ScriptLine(text="Book.", role="cta"))


def _carrying(board, line_index) -> list[Shot]:
    return [s for s in board.shots if line_index in s.line_ids and s.kind == "real"]


def test_a_continuation_takes_a_general_picture_of_the_lines_own_scene_not_a_longer_shot(tmp_path):
    plan = _general_plan(tmp_path)
    ids = _ids(plan)
    board = _board(plan, _pool_script(ids))

    shots = _carrying(board, 1)
    assert len(shots) >= 2 and shots[0].image_id == ids["swimming_pool.jpg"]                  # its own picture first
    for shot in shots[1:]:
        assert shot.image_id in (ids["garden_landscape.jpg"], ids["clubhouse.jpg"])           # a general exterior ...
        assert not set(shot.tags) & {"pool", "gym", "lawn", "lounge"}                         # ... that shows no amenity
        assert shot.seconds <= 4.5
    assert _no_adjacent_repeat(board) and sb.validate_plan(board, plan.brief, plan.assets, plan.image_plan) == []
    assert any(n.startswith(f"The rest of '{LONG_POOL}' plays over ") and "an exterior picture" in n
               and "no other unused photograph shows pool" in n for n in board.notes)


def test_another_photograph_that_shows_what_the_line_needs_comes_first(tmp_path):
    plan = _general_plan(tmp_path, {**GENERAL, "gym.jpg": ["pool"]})                          # a second photograph of a pool
    ids = _ids(plan)
    board = _board(plan, _pool_script(ids))

    shots = _carrying(board, 1)
    assert [s.image_id for s in shots[:2]] == [ids["swimming_pool.jpg"], ids["gym.jpg"]]
    general = [n for n in board.notes if n.startswith("The rest of")]
    assert len(general) == len(shots) - 2 and not any("gym.jpg" in n for n in general)      # only the shots after it


def test_a_photograph_a_later_line_has_claimed_is_never_taken_for_a_continuation(tmp_path):
    tags = {**GENERAL, "gym.jpg": ["pool"]}
    plan = _general_plan(tmp_path, tags)
    ids = _ids(plan)
    later = ScriptLine(text="The kids' pool.", role="feature", tags=["pool"], photo=ids["gym.jpg"])
    board = _board(plan, _pool_script(ids, "swimming_pool.jpg", later))

    first = _carrying(board, 1)
    assert ids["gym.jpg"] not in {s.image_id for s in first}                                 # the other pool is for its line
    assert [s.image_id for s in _carrying(board, 2)][0] == ids["gym.jpg"]


def test_a_continuation_never_takes_another_amenitys_photograph(tmp_path):
    only_amenities = {"exterior_facade.jpg": ["exterior"], "swimming_pool.jpg": ["pool"], "gym.jpg": ["gym"],
                      "garden_landscape.jpg": ["lawn"], "clubhouse.jpg": ["lounge"]}
    plan = _general_plan(tmp_path, only_amenities)
    ids = _ids(plan)
    board = _board(plan, _pool_script(ids))
    shots = _carrying(board, 1)
    assert {s.image_id for s in shots} == {ids["swimming_pool.jpg"]}                          # one shot, joined: nothing general left
    assert shots[0].seconds > 4.5 and any("is joined to the shot before" in n for n in board.notes)


def test_a_weak_tag_makes_a_photograph_a_general_picture_only(tmp_path):
    plan = _general_plan(tmp_path)
    ids = _ids(plan)
    garden = next(a for a in plan.assets if a.filename == "garden_landscape.jpg")
    garden.weak_tags = ["pool"]                                                               # tagged pool, not shown clearly
    board = _board(plan, _pool_script(ids))
    shots = _carrying(board, 1)

    assert shots[0].image_id == ids["swimming_pool.jpg"]                                      # the evidence is the other one
    assert ids["garden_landscape.jpg"] in {s.image_id for s in shots[1:]} or ids["clubhouse.jpg"] in {s.image_id for s in shots[1:]}
    assert "pool" not in next(s for s in board.shots if s.image_id == ids["garden_landscape.jpg"]).tags


def test_a_chip_stays_on_the_first_shot_that_shows_its_tag(tmp_path):
    plan = _general_plan(tmp_path)
    ids = _ids(plan)
    script = _pool_script(ids)
    script.chips, script.chip_lines = ["Swimming pool"], [1]
    board = _board(plan, script)
    chipped = [s for s in board.shots if "Swimming pool" in s.overlay_text]
    assert len(chipped) == 1 and chipped[0].image_id == ids["swimming_pool.jpg"]
    assert not any(n.startswith("Chip 'Swimming pool'") for n in board.notes)


def test_the_hooks_second_shot_may_be_another_exterior(tmp_path):
    plan = _general_plan(tmp_path)
    ids = _ids(plan)
    long_hook = ScriptLine(text="Say hello to a home with everything in the right place, in a city that never stops.",
                           role="hook", topic="exterior")
    board = _board(plan, _script(long_hook, ScriptLine(text="Book.", role="cta")))
    hook = [s for s in board.shots if 0 in s.line_ids and s.kind == "real"]
    assert len(hook) >= 2 and "exterior" in hook[1].tags and hook[1].image_id != hook[0].image_id
    assert _no_adjacent_repeat(board)


def test_no_shot_of_a_plan_is_held_longer_than_the_cap_when_photographs_are_left(tmp_path):
    plan = _general_plan(tmp_path)
    board = _board(plan, _pool_script(_ids(plan)))
    assert max(s.seconds for s in board.shots if s.kind == "real") <= 4.5
    assert not any("held longer than" in n for n in board.notes)


def test_a_plan_with_too_few_photographs_says_which_shots_are_held_too_long(tmp_path):
    plan = Plan(make_video_property(tmp_path, photos=3), {"duration_s": 30})                  # offline: no tags
    board = _board(plan, plan.script)
    long = [s for s in board.shots if s.kind == "real" and s.seconds > 4.5]
    assert long
    note = next(n for n in board.notes if "held longer than 4.5 s" in n)
    assert note.startswith(f"{len(long)} shot(s) are held longer than 4.5 s (the longest, ")
    assert "Add photographs, or shorten the video." in note


def test_a_slow_pace_gets_more_shots_so_that_none_plans_longer_than_the_target(tmp_path):
    plan = Plan(make_video_property(tmp_path, category="Luxury"), {"duration_s": 45})         # calm pace: 3.6 s a shot
    board = _board(plan, plan.script)
    real = [s for s in board.shots if s.kind == "real"]
    joined = [n for n in board.notes if "is joined to the shot before" in n]
    assert len(real) + len(joined) == 13                                                    # 13 shots planned, not 12
    assert sum(1 for s in real if s.seconds > 3.6) == len(joined)                           # only a joined shot is longer
