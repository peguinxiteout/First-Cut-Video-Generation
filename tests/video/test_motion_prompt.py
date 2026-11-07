"""The words sent to an image-to-video model, per model style and camera move."""
from __future__ import annotations

import pytest

from src.agents.video import motion_prompt
from src.data.video_schemas import CAMERA_MOVES
from src.models.video_gen import selector


@pytest.fixture(scope="module")
def catalog():
    return selector.load_catalog()


@pytest.mark.parametrize("camera", CAMERA_MOVES)
def test_h3_gets_positive_words_only_for_every_move(catalog, camera):
    p = motion_prompt.for_provider(catalog["minimax_h3_max"].params, camera, scene="exterior")
    assert p.style == "positive_only" and p.negative_prompt == ""
    assert "The scene stays exactly as photographed" in p.prompt
    assert motion_prompt.negations(p.prompt) == []


@pytest.mark.parametrize("camera", CAMERA_MOVES)
def test_kling_gets_the_same_rule_and_a_negative_prompt(catalog, camera):
    p = motion_prompt.for_provider(catalog["kling_25_turbo_pro"].params, camera, scene="interior")
    assert p.style == "with_negative"
    assert "The scene stays exactly as photographed" in p.prompt
    for words in ("new people", "new objects", "text", "changes to the building", "morphing"):
        assert words in p.negative_prompt


@pytest.mark.parametrize("camera,words", [
    ("push_in", "push-in towards the building"), ("pull_out", "pull-back from the building"),
    ("pan_left", "pan to the left"), ("pan_right", "pan to the right"), ("tilt_up", "tilt upwards"),
    ("tilt_down", "tilt downwards"), ("still", "holds still"),
])
def test_the_move_is_said_first(camera, words):
    p = motion_prompt.build(camera, scene="exterior")
    assert p.prompt.startswith("Slow, steady camera" if camera != "still" else "The camera holds still")
    assert words in p.prompt


def test_the_subject_follows_the_scene():
    assert "towards the room" in motion_prompt.build("push_in", scene="interior").prompt
    assert "towards the amenity" in motion_prompt.build("push_in", scene="amenity").prompt
    assert "towards the scene" in motion_prompt.build("push_in", scene="unknown").prompt
    assert "towards the clubhouse" in motion_prompt.build("push_in", subject="the clubhouse").prompt


def test_a_transition_is_one_continuous_move_between_two_real_views(catalog):
    p = motion_prompt.for_provider(catalog["minimax_h3_max"].params, transition=True)
    assert p.prompt.startswith("One continuous camera move from the first view to the second; "
                               "both places stay exactly as photographed.")
    assert motion_prompt.negations(p.prompt) == []
    k = motion_prompt.for_provider(catalog["kling_25_turbo_pro"].params, transition=True)
    assert k.prompt == p.prompt and k.negative_prompt


def test_an_unknown_move_is_refused():
    with pytest.raises(ValueError, match="unknown camera move"):
        motion_prompt.build("orbit")


def test_a_negation_in_a_positive_only_prompt_is_refused(monkeypatch):
    real = motion_prompt._cfg()
    monkeypatch.setattr(motion_prompt, "_cfg", lambda: {**real, "shot": "$move No people are added."})
    with pytest.raises(ValueError, match="negation"):
        motion_prompt.build("push_in")
    assert motion_prompt.build("push_in", takes_negative=True).negative_prompt


def test_negations_are_whole_words():
    assert motion_prompt.negations("Nothing changes, not now") == ["nothing", "not"]
    assert motion_prompt.negations("a notable knot in the north") == []
