"""Node 4: the offline script, and the guard on everything it says and shows."""
from __future__ import annotations

import pytest

from src.agents.brochure import facts, strategy
from src.agents.video import script as script_agent
from src.agents.video.guard import VideoGuard
from src.data.video_schemas import ScriptLine, VideoScript
from tests.video.helpers import Plan, make_video_property


def _guard(plan: Plan) -> VideoGuard:
    return VideoGuard(plan.prop, plan.knowledge, facts.build(plan.prop, plan.brief.category),
                      strategy.for_category(plan.brief.category))


def _plan(tmp_path, category="Mid-Tier", **overrides) -> Plan:
    return Plan(make_video_property(tmp_path, category), overrides or None)


def test_the_script_opens_with_a_short_hook_and_closes_with_the_call_to_action(tmp_path):
    plan = _plan(tmp_path)
    lines = plan.script.lines

    assert [l.role for l in lines][0] == "hook" and lines[-1].role == "cta"
    assert lines[0].words <= 9 and plan.script.mode == "template"
    assert "Book a site visit" in lines[-1].text
    assert not [n for n in plan.script.notes if n.startswith("Check:")]


def test_the_words_fit_the_length_within_ten_percent(tmp_path):
    plan = _plan(tmp_path)
    target = plan.brief.target_words
    assert target == 27
    assert target * 0.9 <= plan.script.word_count <= target * 1.1
    assert plan.script.word_count == sum(l.words for l in plan.script.lines)


@pytest.mark.parametrize("seconds, roles", [
    (10, ["hook", "feature", "cta"]),
    (15, ["hook", "feature", "feature", "cta"]),
])
def test_the_lines_follow_the_structure_for_the_length(tmp_path, seconds, roles):
    plan = _plan(tmp_path, duration_s=seconds)
    assert [l.role for l in plan.script.lines][:len(roles) - 1] == roles[:-1]
    assert plan.script.lines[-1].role == "cta"
    assert plan.script.word_count <= plan.brief.target_words * 1.1


def test_a_longer_video_gets_proof_lines_too(tmp_path):
    plan = _plan(tmp_path, duration_s=30)
    assert "proof" in [l.role for l in plan.script.lines]


def test_every_line_and_every_piece_of_screen_text_passes_the_guard(tmp_path):
    plan = _plan(tmp_path, duration_s=30)
    guard = _guard(plan)
    s = plan.script
    for line in s.lines:
        assert guard.why(line.text, repeat=False) == "", line.text
    for text in [*s.on_screen, *s.chips, s.cta, *s.end_card]:
        assert guard.why(text, repeat=False) == "", text


def test_the_same_property_gets_the_same_script_every_time(tmp_path):
    first = _plan(tmp_path / "a")
    second = _plan(tmp_path / "b")
    assert [l.text for l in first.script.lines] == [l.text for l in second.script.lines]


def test_highlights_come_first_and_keep_their_own_words(tmp_path):
    plan = _plan(tmp_path, highlights=["Rooftop infinity pool", "Zero-waste community"])
    features = [l for l in plan.script.lines if l.role == "feature"]
    assert [f.text for f in features][:2] == ["Rooftop infinity pool.", "Zero-waste community."]
    assert features[0].topic == "pool"


def test_a_highlight_with_an_invented_number_is_dropped_and_the_reason_recorded(tmp_path):
    plan = _plan(tmp_path, highlights=["50 acres of landscaped gardens", "Rooftop infinity pool"])

    assert not any("50" in l.text for l in plan.script.lines)
    assert "Rooftop infinity pool." in [l.text for l in plan.script.lines]
    assert ("Highlight dropped: '50 acres of landscaped gardens' because '50' is not a figure in the "
            "property's facts.") in plan.script.notes


def test_a_highlight_with_a_number_word_is_dropped_too(tmp_path):
    plan = _plan(tmp_path, highlights=["Twenty acres of gardens"])
    assert not any("wenty" in l.text for l in plan.script.lines)
    assert any("'twenty' is not a figure" in n for n in plan.script.notes)


def test_a_highlight_whose_figure_is_a_fact_is_kept(tmp_path):
    plan = _plan(tmp_path, highlights=["Possession in December 2027"])
    assert "Possession in December 2027." in [l.text for l in plan.script.lines]
    assert not [n for n in plan.script.notes if n.startswith("Highlight dropped")]


def test_a_highlight_with_a_damaged_character_is_dropped(tmp_path):
    plan = _plan(tmp_path, highlights=["Sports caf� and lounge"])
    assert not any("�" in l.text for l in plan.script.lines)
    assert any("damaged character" in n for n in plan.script.notes)


def test_a_highlight_the_register_avoids_is_dropped(tmp_path):
    plan = _plan(tmp_path, highlights=["A world-class address"])
    assert any("register avoids" in n for n in plan.script.notes)


def test_short_selling_points_of_the_property_are_used_as_features(tmp_path):
    texts = [l.text for l in _plan(tmp_path).script.lines]
    assert "Two minutes from the Baner metro station." in texts       # its own number word is a fact there


def test_amenities_become_lines_with_the_topic_a_photograph_can_match(tmp_path):
    plan = _plan(tmp_path, highlights=["Rooftop infinity pool"] * 0, duration_s=30)
    zone_lines = [l for l in plan.script.lines if l.text.startswith(("On site:", "Right at home:"))]
    assert zone_lines and {l.topic for l in zone_lines} <= {"pool", "fitness", "clubhouse", "kids", "sports",
                                                            "landscape", "security", "convenience", "care"}
    assert any(l.topic == "pool" for l in zone_lines)


def test_the_price_is_on_screen_only_for_a_category_that_shows_it(tmp_path):
    mid = _plan(tmp_path / "m", "Mid-Tier").script.on_screen
    luxury = _plan(tmp_path / "l", "Luxury").script.on_screen
    assert mid[0] == "Aurelia Heights" and mid[1].startswith("From ") and "85 Lakh" in mid[1]
    assert luxury == ["Aurelia Heights"]


def test_there_are_at_most_two_chips_and_each_belongs_to_a_feature_line(tmp_path):
    plan = _plan(tmp_path, highlights=["Rooftop infinity pool", "Zero-waste community", "Yoga deck"], duration_s=30)
    s = plan.script
    assert 1 <= len(s.chips) <= 2 and len(s.chips) == len(s.chip_lines)
    assert all(s.lines[i].role == "feature" for i in s.chip_lines)
    assert s.chips[0] == "Rooftop infinity pool"


def test_the_end_card_carries_rera_and_contact_and_the_call_to_action_is_the_briefs(tmp_path):
    s = _plan(tmp_path, cta="Visit this weekend").script
    assert s.cta == "Visit this weekend"
    assert s.end_card == ["RERA: P52100012345", "+91 98765 43210", "https://northlight.example"]


def test_a_property_with_no_rera_number_says_the_card_shows_none(tmp_path):
    plan = Plan(make_video_property(tmp_path, rera_number=""), None)
    assert not any(t.startswith("RERA") for t in plan.script.end_card)
    assert "No RERA number is given for this property, so the closing card shows none." in plan.script.notes
    with_rera = _plan(tmp_path / "x")
    assert not any("No RERA number" in n for n in with_rera.script.notes)


def test_a_call_to_action_with_an_invented_figure_is_replaced_by_the_default(tmp_path):
    s = _plan(tmp_path, cta="Book now and save 40 percent").script
    assert s.cta == "Discover more"
    assert any(n.startswith("Call to action dropped") for n in s.notes)


def test_too_long_a_script_is_trimmed_from_the_end(tmp_path):
    long = "A landscaped courtyard with seating and shade trees where neighbours meet every evening"
    plan = _plan(tmp_path, highlights=[long, long.replace("courtyard", "terrace")])
    assert plan.script.word_count <= plan.brief.target_words * 1.1
    assert any(n.startswith("Trimmed to fit 15 s") for n in plan.script.notes)


def test_too_short_a_script_says_so(tmp_path):
    plan = Plan(make_video_property(tmp_path, usps="", amenities="Parking"), None)
    assert plan.script.word_count < plan.brief.target_words * 0.9
    assert any("words against a target of 27" in n for n in plan.script.notes)


def test_narration_says_and_for_a_slash_but_the_chip_keeps_it(tmp_path):
    plan = Plan(make_video_property(tmp_path, usps="", amenities="Tennis/pickleball court\nSwimming pool"),
                {"duration_s": 30})
    assert not any("/" in l.text for l in plan.script.lines)
    assert any("tennis and pickleball court" in l.text for l in plan.script.lines)
    assert "Tennis/pickleball court" in plan.script.chips


def test_a_highlight_with_a_slash_is_spoken_with_and(tmp_path):
    plan = _plan(tmp_path, highlights=["Yoga/pilates studio"])
    assert "Yoga and pilates studio." in [l.text for l in plan.script.lines]
    assert plan.script.chips[0] == "Yoga/pilates studio"


def test_no_two_lines_in_a_row_begin_the_same_way(tmp_path):
    for seconds in (30, 45, 60):
        lines = _plan(tmp_path / str(seconds), duration_s=seconds).script.lines
        openers = [" ".join(l.text.split()[:2]).lower() for l in lines]
        assert all(a != b for a, b in zip(openers, openers[1:])), openers


def test_the_opener_of_a_frame_is_its_words_before_the_first_slot():
    assert script_agent._opener("On site: {item_list}.") == "On site:"
    assert script_agent._opener("{item_list}, all on site.") == "{item_list}"
    assert script_agent._opener("[has:pool] Right at home: {item_list}.") == "Right at home:"


def test_a_frame_that_begins_like_the_last_line_is_used_only_when_nothing_else_will_do(tmp_path):
    plan = _plan(tmp_path, duration_s=45)
    writer = script_agent._Writer(plan.prop, plan.brief, plan.knowledge)
    frames = ["On site: {item_list}.", "On site: {item_list}, too."]
    values = {**writer.values, "item_list": "swimming pool"}
    first = writer.pick(frames, values)
    second = writer.pick(frames, {**values, "item_list": "gymnasium"})
    assert first.startswith("On site:") and second.startswith("On site:")      # only one opener exists: still filled
    third = writer.pick(["On site: {item_list}.", "Right here: {item_list}."], {**values, "item_list": "yoga lawn"})
    assert third.startswith("Right here:")


def test_the_tone_changes_the_wording(tmp_path):
    texts = {tone: [l.text for l in Plan(make_video_property(tmp_path / tone), {"tone": tone}).script.lines]
             for tone in ("warm", "confident", "refined")}
    assert len({tuple(t) for t in texts.values()}) == 3


def test_the_objective_changes_the_opening_and_the_close(tmp_path):
    site = Plan(make_video_property(tmp_path / "s", cta=""), {"objective": "site_visit"}).script.lines
    enquiry = Plan(make_video_property(tmp_path / "e", cta=""), {"objective": "enquiry"}).script.lines
    assert site[0].text != enquiry[0].text
    assert "site visit" in site[-1].text.lower() and "enquire" in enquiry[-1].text.lower()


def test_the_concept_carries_pace_music_and_the_budget(tmp_path):
    concept = _plan(tmp_path, "Luxury").result.concept
    assert concept["pace"] == "calm" and concept["shot_s"] == 3.6 and concept["tone"] == "refined"
    assert concept["music_mood"] == "calm cinematic" and concept["voice"] == "default"
    assert concept["target_words"] == 27 and concept["structure"][0] == "hook"


def test_the_self_check_names_what_is_wrong():
    ok = VideoScript(lines=[ScriptLine(text="Welcome home.", role="hook"), ScriptLine(text="Visit.", role="cta")])
    assert script_agent.check(ok, 9) == []
    long_hook = VideoScript(lines=[ScriptLine(text="one two three four five six seven eight nine ten", role="hook"),
                                   ScriptLine(text="Visit.", role="cta")])
    assert script_agent.check(long_hook, 9) == ["the hook is 10 words; the limit is 9"]
    assert script_agent.check(VideoScript(), 9) == ["the script has no lines"]
    assert "the first line is not the hook" in script_agent.check(
        VideoScript(lines=[ScriptLine(text="Hi.", role="feature"), ScriptLine(text="Go.", role="cta")]), 9)
    assert "the last line is not the call to action" in script_agent.check(
        VideoScript(lines=[ScriptLine(text="Hi.", role="hook")]), 9)
