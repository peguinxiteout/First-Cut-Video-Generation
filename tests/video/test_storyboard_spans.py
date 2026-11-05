"""A line that spans two shots uses two different photographs, or one longer shot; the same photograph is never in two
shots one after the other."""
from __future__ import annotations

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
