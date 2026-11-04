"""Node 5: the rule planner, and the check any plan has to pass."""
from __future__ import annotations

import pytest

from src.agents.brochure import facts, image_plan, strategy
from src.agents.video import storyboard as sb
from src.agents.video.guard import VideoGuard
from src.core.exceptions import WorkflowError
from src.data.video_schemas import CAMERA_MOVES, RankedAsset, ScriptLine, Shot, Storyboard, VideoScript
from tests.video.helpers import Plan, make_video_property


def _plan(tmp_path, category="Mid-Tier", photos=8, **overrides) -> Plan:
    return Plan(make_video_property(tmp_path, category, photos=photos), overrides or None)


def _board(plan: Plan) -> Storyboard:
    return sb.run(plan.brief, plan.script, plan.assets, plan.analyses, plan.image_plan)


def _problems(board, plan, **kwargs):
    return sb.validate_plan(board, plan.brief, plan.assets, plan.image_plan, **kwargs)


@pytest.fixture
def plan(tmp_path) -> Plan:
    return _plan(tmp_path)


@pytest.fixture
def board(plan) -> Storyboard:
    return _board(plan)


# -- the plan ---------------------------------------------------------------------------------

def test_the_shots_add_up_to_the_length_and_end_on_a_two_second_card(plan, board):
    assert board.total_seconds == pytest.approx(15.0, abs=0.05)
    last = board.shots[-1]
    assert (last.kind, last.seconds, last.camera) == ("card", 2.0, "still")
    assert len(board.shots) == plan.brief.n_shots
    assert [s.kind for s in board.shots[:-1]] == ["real"] * (plan.brief.n_shots - 1)
    assert board.planner_mode == "rules" and _problems(board, plan) == []


@pytest.mark.parametrize("seconds", [10, 30])
def test_other_lengths_add_up_too(tmp_path, seconds):
    plan = _plan(tmp_path, duration_s=seconds)
    board = _board(plan)
    assert board.total_seconds == pytest.approx(seconds, abs=0.05)
    assert _problems(board, plan) == []


def test_the_hook_is_the_best_exterior_and_gets_at_least_three_seconds(plan, board):
    hook = board.shots[0]
    assert hook.subject == "exterior" and hook.source_path.endswith("exterior_facade.jpg")
    assert hook.seconds >= 3.0 and hook.camera == "push_in"
    assert hook.overlay_text == plan.script.on_screen


def test_no_photograph_or_twin_is_shown_twice_when_there_are_enough(plan, board):
    ids = [s.image_id for s in board.shots if s.kind == "real"]
    assert len(ids) == len(set(ids))
    clusters = [plan.image_plan.cluster(i) for i in ids]
    assert len(clusters) == len(set(clusters))
    assert not any(s.reuse for s in board.shots)


def test_a_line_about_something_gets_a_photograph_of_it(tmp_path):
    plan = _plan(tmp_path, duration_s=30)
    board = _board(plan)
    matched = 0
    for shot in board.shots[1:-1]:
        wanted = next((plan.script.lines[i].topic for i in shot.line_ids if plan.script.lines[i].topic), "")
        matched += bool(wanted) and shot.subject == wanted      # two lines on one topic share one photograph
    assert matched >= 2
    assert any("show what their line talks about" in n for n in board.notes)


def test_every_script_line_is_under_some_shot_and_the_call_to_action_is_on_the_card(plan, board):
    covered = {i for s in board.shots for i in s.line_ids}
    assert covered == set(range(len(plan.script.lines)))
    cta = [i for i, l in enumerate(plan.script.lines) if l.role == "cta"]
    assert board.shots[-1].line_ids == cta


def test_cameras_are_known_never_repeat_back_to_back_and_each_has_a_focal_point(plan, board):
    real = [s for s in board.shots if s.kind == "real"]
    assert all(s.camera in CAMERA_MOVES for s in board.shots)
    assert all(a.camera != b.camera for a, b in zip(real, real[1:]))
    assert all(len(s.focal.split()) == 2 and s.focal.replace("%", "").replace(" ", "").replace(".", "").isdigit()
               for s in real)
    assert all(s.motion_prompt.endswith("Add no people, objects or text.") for s in real)


def test_a_picture_wider_than_the_frame_pans_and_a_taller_one_tilts(tmp_path):
    plan = _plan(tmp_path, duration_s=30)
    board = _board(plan)
    by_id = {a.image_id: a for a in plan.assets}
    for shot in board.shots[1:-1]:
        wide = by_id[shot.image_id].aspect > plan.brief.width / plan.brief.height * 1.3
        pans = {"pan_left", "pan_right", "push_in", "pull_out"}
        tilts = {"tilt_up", "tilt_down", "push_in", "pull_out"}
        assert shot.camera in (pans if wide else tilts)


def test_chips_show_with_their_own_line_and_only_once(tmp_path):
    plan = _plan(tmp_path, highlights=["Rooftop infinity pool"], duration_s=15)
    board = _board(plan)
    shown = [t for s in board.shots[1:-1] for t in s.overlay_text]
    assert shown == plan.script.chips[:len(shown)] and len(shown) == len(set(shown))
    chip_shot = next(s for s in board.shots if "Rooftop infinity pool" in s.overlay_text)
    assert plan.script.chip_lines[0] in chip_shot.line_ids


def test_a_chip_is_not_printed_over_a_photograph_not_known_to_show_it(tmp_path):
    plan = _plan(tmp_path, highlights=["Rooftop infinity pool"])
    for asset in plan.assets:
        asset.topic = "lifestyle"                      # what an offline reading of generic file names gives
    board = _board(plan)
    assert [t for s in board.shots[1:-1] for t in s.overlay_text] == []
    assert any(n.startswith("Chip 'Rooftop infinity pool' not shown: no photograph under its line") for n in board.notes)
    assert board.shots[0].overlay_text[:len(plan.script.on_screen)] == plan.script.on_screen


def test_a_chip_is_printed_when_the_photograph_matches_its_topic(tmp_path):
    plan = _plan(tmp_path, highlights=["Rooftop infinity pool"])
    board = _board(plan)
    chip_shot = next(s for s in board.shots if "Rooftop infinity pool" in s.overlay_text)
    assert chip_shot.subject == "pool"
    assert not any(n.startswith("Chip 'Rooftop infinity pool' not shown") for n in board.notes)


def test_the_closing_card_prints_the_call_to_action_rera_and_contact(plan, board):
    assert board.shots[-1].overlay_text == [plan.script.cta, *plan.script.end_card]


def test_the_rule_planner_never_makes_a_generated_shot(tmp_path):
    plan = _plan(tmp_path, allow_generated=True, creative_direction="a lifestyle scene")
    board = _board(plan)
    assert plan.brief.allow_generated and board.generated_shots == []
    assert all(s.image_prompt == "" for s in board.shots)


# -- the hero image ----------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["swimming_pool.jpg", "SWIMMING_POOL.JPG", "swimming_pool"])
def test_the_hero_image_opens_the_video_by_file_name_with_or_without_the_extension(tmp_path, name):
    plan = _plan(tmp_path, hero_image=name)
    board = _board(plan)
    assert board.shots[0].source_path.endswith("swimming_pool.jpg") and board.shots[0].subject == "pool"
    shown = [s.image_id for s in board.shots if s.kind == "real"]
    assert len(shown) == len(set(shown))                           # and it is not shown again
    assert not any("hero_image" in n for n in board.notes) and _problems(board, plan) == []


def test_the_hero_image_can_be_named_by_image_id(tmp_path):
    plan = _plan(tmp_path)
    pool = next(a for a in plan.assets if a.filename == "swimming_pool.jpg")
    plan.brief.hero_image = pool.image_id.lower()
    assert _board(plan).shots[0].image_id == pool.image_id


def test_a_hero_image_that_does_not_exist_is_ignored_with_a_note(tmp_path):
    plan = _plan(tmp_path, hero_image="nothing.jpg")
    board = _board(plan)
    assert board.shots[0].subject == "exterior"                    # the planner's own choice
    assert "hero_image 'nothing.jpg' matches no uploaded picture; ignored." in board.notes


def test_a_hero_image_that_cannot_carry_a_shot_is_ignored_with_the_reason(tmp_path):
    plan = _plan(tmp_path, hero_image="floorplan_plan_0.png")
    board = _board(plan)
    assert board.shots[0].subject == "exterior"
    assert "hero_image 'floorplan_plan_0.png' cannot be used (a floorplan, not a photograph); ignored." in board.notes


# -- too few photographs -----------------------------------------------------------------------

def test_with_fewer_photographs_than_shots_one_is_shown_again_from_the_other_side(tmp_path):
    plan = _plan(tmp_path, photos=2)
    board = _board(plan)

    real = [s for s in board.shots if s.kind == "real"]
    again = [s for s in real if s.reuse]
    assert len(real) == 3 and len(again) == 1
    first = next(s for s in real if s.image_id == again[0].image_id and not s.reuse)
    assert again[0].camera != first.camera and again[0].focal != first.focal
    assert any("again, cropped from the other side" in n for n in board.notes)
    assert board.total_seconds == pytest.approx(15.0, abs=0.05) and _problems(board, plan) == []


def test_a_reused_photograph_never_follows_itself(tmp_path):
    board = _board(_plan(tmp_path, photos=3, duration_s=30))
    real = [s for s in board.shots if s.kind == "real"]
    assert all(a.image_id != b.image_id for a, b in zip(real, real[1:]))


def test_a_single_photograph_makes_one_long_shot_not_a_stutter(tmp_path):
    plan = _plan(tmp_path, photos=1)
    board = _board(plan)
    assert [s.kind for s in board.shots] == ["real", "card"]
    assert board.shots[0].seconds == pytest.approx(13.0, abs=0.05) and not board.shots[0].reuse
    assert any("allow 1 photograph shot(s), not 4" in n for n in board.notes)


def test_with_no_usable_photograph_the_video_is_cards_only(tmp_path):
    plan = _plan(tmp_path, photos=0)
    board = _board(plan)
    assert [s.kind for s in board.shots] == ["card", "card"]
    assert board.shots[0].overlay_text[:len(plan.script.on_screen)] == plan.script.on_screen
    assert "No usable photograph: the video is title cards only." in board.notes
    assert board.total_seconds == pytest.approx(15.0, abs=0.05) and _problems(board, plan) == []


# -- the timeline helpers -----------------------------------------------------------------------

def test_photo_seconds_put_the_hook_first_and_add_up():
    seconds = sb.photo_seconds(15, 4, 2.0, 3.0)
    assert seconds[0] >= 3.0 and len(seconds) == 4 and sum(seconds) == pytest.approx(13.0, abs=0.01)
    assert sb.photo_seconds(15, 1, 2.0, 3.0) == [13.0]
    short = sb.photo_seconds(10, 3, 2.0, 3.0)
    assert short[0] == 3.0 and sum(short) == pytest.approx(8.0, abs=0.01)


def test_lines_are_laid_on_the_shots_by_where_their_middle_falls():
    script = VideoScript(lines=[ScriptLine(text="word " * 4, role="hook"), ScriptLine(text="word " * 8, role="feature"),
                                ScriptLine(text="word " * 4, role="feature"), ScriptLine(text="go now", role="cta")])
    ids = sb.assign_lines(script, [4.0, 4.0, 4.0], words_per_second=2.0)
    assert ids[0] == [0] and ids[-1] == [2] and all(ids)
    assert 3 not in [i for group in ids for i in group]                       # the card takes the call to action


def test_a_shot_with_no_line_middle_carries_on_the_line_playing():
    script = VideoScript(lines=[ScriptLine(text="word " * 20, role="hook"), ScriptLine(text="end", role="cta")])
    ids = sb.assign_lines(script, [3.0, 3.0, 3.0], words_per_second=2.0)
    assert ids == [[0], [0], [0]]


# -- validate_plan --------------------------------------------------------------------------------

def test_a_good_plan_has_no_problems(plan, board):
    assert _problems(board, plan) == []


def test_an_empty_storyboard_is_a_problem(plan):
    assert _problems(Storyboard(), plan) == ["the storyboard has no shots"]


def test_a_photograph_that_does_not_exist_is_named(plan, board):
    board = board.model_copy(deep=True)
    board.shots[1].image_id = "IMG999"
    assert any("Shot 2 uses a photograph that does not exist (IMG999)" in p for p in _problems(board, plan))


def test_a_drawing_or_a_logo_cannot_carry_a_shot(plan, board):
    board = board.model_copy(deep=True)
    plan_asset = next(a for a in plan.assets if not a.usable and a.topic == "floorplan")
    board.shots[1].image_id = plan_asset.image_id
    problems = _problems(board, plan)
    assert any(f"uses {plan_asset.filename}, which is not usable (a floorplan, not a photograph)" in p for p in problems)


def test_showing_a_photograph_again_needs_the_reuse_flag(plan, board):
    board = board.model_copy(deep=True)
    board.shots[2].image_id = board.shots[1].image_id
    assert any("again without being marked reuse" in p for p in _problems(board, plan))
    board.shots[2].reuse = True
    assert not any("without being marked reuse" in p for p in _problems(board, plan))


def test_the_reuse_flag_on_a_first_showing_is_a_problem(plan, board):
    board = board.model_copy(deep=True)
    board.shots[1].reuse = True
    assert any("is marked reuse but is the first showing" in p for p in _problems(board, plan))


def test_two_pictures_that_look_alike_count_as_one():
    a = RankedAsset(image_id="A", filename="a.jpg", path="a", cluster=0)
    b = RankedAsset(image_id="B", filename="b.jpg", path="b", cluster=0)
    twins = image_plan.Plan(cluster_of={"A": 0, "B": 0}, members={0: ["A", "B"]})
    brief = _brief()
    board = Storyboard(shots=[Shot(index=0, image_id="A", seconds=6.5), Shot(index=1, image_id="B", seconds=6.5),
                              Shot(index=2, kind="card", camera="still", seconds=2.0)])
    problems = sb.validate_plan(board, brief, [a, b], twins)
    assert any("Shot 2 shows b.jpg (or a picture like it) again" in p for p in problems)


def test_the_shots_must_add_up_to_the_length(plan, board):
    board = board.model_copy(deep=True)
    board.shots[0].seconds += 2
    assert any(p.startswith("the shots add up to 17.0 s, not 15 s") for p in _problems(board, plan))


def test_an_unknown_camera_move_or_kind_is_a_problem(plan, board):
    board = board.model_copy(deep=True)
    board.shots[1].camera = "whip_pan"
    board.shots[2].kind = "text_to_video"
    problems = _problems(board, plan)
    assert "Shot 2 has an unknown camera move 'whip_pan'" in problems
    assert "Shot 3 has an unknown kind 'text_to_video'" in problems


def test_a_shot_with_no_length_is_a_problem(plan, board):
    board = board.model_copy(deep=True)
    board.shots[1].seconds = 0
    assert "Shot 2 has no length" in _problems(board, plan)


def test_the_video_must_end_on_a_card(plan, board):
    board = board.model_copy(deep=True)
    board.shots[-1].kind = "real"
    assert "the video does not end on a closing card" in _problems(board, plan)


class TestGeneratedShots:
    def _with_generated(self, board, **fields):
        board = board.model_copy(deep=True)
        board.shots[1].kind = "generated"
        board.shots[1].image_id = ""
        for name, value in fields.items():
            setattr(board.shots[1], name, value)
        return board

    def test_not_allowed_unless_the_brief_allows_it(self, plan, board):
        generated = self._with_generated(board, generated_kind="lifestyle", image_prompt="a family at dusk")
        assert any("does not allow generated pictures" in p for p in _problems(generated, plan))

    def test_allowed_kinds_come_from_the_sectors_style_file(self, plan, board):
        assert sb.allowed_generated_kinds() == ["lifestyle", "amenity_concept", "neighbourhood"]
        plan.brief.allow_generated = True
        for kind in sb.allowed_generated_kinds():
            generated = self._with_generated(board, generated_kind=kind, image_prompt="a scene")
            assert _problems(generated, plan) == [], kind

    def test_the_building_or_the_flats_are_never_allowed(self, plan, board):
        plan.brief.allow_generated = True
        for kind in ("building", "flat_interior", "view_from_flat", ""):
            generated = self._with_generated(board, generated_kind=kind, image_prompt="a scene")
            assert any("only lifestyle, amenity_concept, neighbourhood are allowed" in p
                       for p in _problems(generated, plan)), kind

    def test_a_generated_shot_needs_a_prompt(self, plan, board):
        plan.brief.allow_generated = True
        generated = self._with_generated(board, generated_kind="lifestyle", image_prompt="")
        assert any("has no image prompt" in p for p in _problems(generated, plan))


def test_overlay_text_is_put_through_the_guard_when_one_is_given(plan, board):
    guard = VideoGuard(plan.prop, plan.knowledge, facts.build(plan.prop, "mid"), strategy.for_category("mid"))
    assert _problems(board, plan, guard=guard) == []
    board = board.model_copy(deep=True)
    board.shots[1].overlay_text = ["Only 12 homes left"]
    assert any("prints 'Only 12 homes left', but '12' is not a figure" in p
               for p in _problems(board, plan, guard=guard))


def test_a_planner_that_breaks_its_own_rules_fails_loudly(plan, monkeypatch):
    monkeypatch.setattr(sb, "validate_plan", lambda *a, **k: ["something is wrong"])
    with pytest.raises(WorkflowError, match="fails its own checks: something is wrong"):
        _board(plan)


def _brief():
    from src.data.video_schemas import VideoBrief

    return VideoBrief(property_id="P", name="P", duration_s=15)
