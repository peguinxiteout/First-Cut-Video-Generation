"""The Video tab: the prompt box and its toolbar, Plan off and Plan on, the model menu, the estimate, progress,
approval and the result. Jobs and the prompt reader are stubbed; the database and the data folder are in a temp
folder (the app calls init_db on the database in Settings, so database_path is moved before every run).
AppTest drives the widgets inside the toolbar's popovers like any other, by key."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.core.config import settings  # noqa: E402
from src.core.exceptions import IngestionError, JobBusyError, JobError  # noqa: E402
from src.agents.video import prompt_brief  # noqa: E402
from src.agents.video.prompt_brief import PromptBriefResult  # noqa: E402
from src.data import video_repository as repository  # noqa: E402
from src.data.schemas import AgentStepResult  # noqa: E402
from src.data.video_schemas import (  # noqa: E402
    PLAN_STATE_FILE, PromptBrief, VideoEvaluationResult, VideoGate, VideoGenerationRecord, VideoInput, VideoRequest,
    VideoSummary,
)
from src.models.video_gen import registry  # noqa: E402
from src.models.video_gen.mock import MockVideoProvider  # noqa: E402
from src.ui import video_box, video_tab  # noqa: E402
from src.workflows import video as workflow  # noqa: E402
from src.workflows import video_jobs  # noqa: E402
from tests.video.helpers import make_video_property  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "streamlit_app.py")
PROPERTY = "AURELIA-HEIGHTS"


@pytest.fixture(autouse=True)
def _database(tmp_path, monkeypatch):
    """Not the real data/app.db: the app opens whatever Settings says."""
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "ui.db"), raising=False)
    assert settings.database_path_resolved.parent == tmp_path
    assert settings.data_dir_resolved == (tmp_path / "data").resolve()


@pytest.fixture(autouse=True)
def refuse_models(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a model was asked")

    for target in ("src.agents.brochure.property_intelligence.call_with_fallback",
                   "src.agents.brochure.vision.call_with_fallback"):
        monkeypatch.setattr(target, refuse)


@pytest.fixture
def folder(tmp_path):
    return make_video_property(tmp_path / "data" / "properties")


@pytest.fixture
def calls(monkeypatch):
    """start_plan, start_produce and cancel are stubs that record what they were asked."""
    seen = {"plan": [], "produce": [], "cancel": []}

    def start_plan(property_id, overrides=None, **flags):
        seen["plan"].append((property_id, overrides, flags))
        return "vid_stub000001"

    monkeypatch.setattr(video_jobs, "start_plan", start_plan)
    monkeypatch.setattr(video_jobs, "start_produce", lambda video_id: seen["produce"].append(video_id))
    monkeypatch.setattr(video_jobs, "cancel", lambda video_id: seen["cancel"].append(video_id) or True)
    return seen


def _app(**query) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["solution"] = "Video"
    for name, value in query.items():
        at.query_params[name] = value
    return at


def _key(field: str) -> str:
    return video_tab.form_key(PROPERTY, field)


def _shown(at) -> str:
    parts = []
    for kind in ("markdown", "caption", "error", "warning", "success", "info", "subheader", "title"):
        parts += [str(e.value) for e in getattr(at, kind)]
    parts += [c.label for c in at.checkbox]
    return "\n".join(parts)


def _seed(video_id="vid_seed000001", status="planning", **fields) -> VideoGenerationRecord:
    repository.init_db()
    record = VideoGenerationRecord(video_id=video_id, property_id=PROPERTY, status=status, **fields)
    repository.save_generation(record)
    return record


# -- the pure parts ---------------------------------------------------------------------------------------

def test_every_option_is_sent_and_a_cleared_field_is_sent_empty():
    values = {"duration_s": 10, "aspect": "16:9", "audience": "   ", "objective": "site_visit",
              "highlights": "", "tone": "", "voice": "male", "music_mood": "", "cta": "", "creative_direction": "",
              "hero_image": "", "burn_captions": False, "grade": False, "allow_generated": False,
              "pronunciations": "", "model": "kenburns"}
    overrides = video_tab.build_overrides(values)

    assert set(overrides) == set(VideoInput.model_fields)
    assert overrides["audience"] == "" and overrides["highlights"] == [] and overrides["pronunciations"] == {}
    assert overrides["burn_captions"] is False and overrides["grade"] is False
    assert (overrides["duration_s"], overrides["aspect"], overrides["objective"]) == (10, "16:9", "site_visit")
    VideoInput.model_validate(overrides)                       # what the form sends is a valid set of options


def test_lines_become_lists_and_pairs():
    overrides = video_tab.build_overrides({
        "highlights": "  • Swimming Pool\n- Gym\n\nSwimming Pool\n   Clubhouse  ",
        "pronunciations": "Anamika = Ah-nah-mih-kah\nbad line\nSG Highway=S G Highway\n=nothing\nword="})
    assert overrides["highlights"] == ["Swimming Pool", "Gym", "Clubhouse"]
    assert overrides["pronunciations"] == {"Anamika": "Ah-nah-mih-kah", "SG Highway": "S G Highway"}
    assert video_tab.parse_pronunciations("a=b\nbad line\n=x\ny=")[1] == ["bad line", "=x", "y="]
    assert video_tab.build_overrides({})["duration_s"] == 15 and video_tab.build_overrides({})["model"] == "auto"


def test_a_cleared_field_clears_what_the_file_says(folder):
    (folder / "video_input.json").write_text(json.dumps({"audience": "families", "highlights": ["Gym"],
                                                        "tone": "warm"}), encoding="utf-8")
    from src.agents.video import brief as brief_agent

    defaults, notes = video_tab.form_defaults(folder)
    assert notes == [] and defaults["audience"] == "families" and defaults["highlights"] == "Gym"
    cleared = {**defaults, "audience": "", "highlights": "", "tone": ""}
    chosen = brief_agent.load_video_input(folder, video_tab.build_overrides(cleared))
    assert (chosen.audience, chosen.highlights, chosen.tone) == ("", [], "")


def test_the_form_starts_from_the_defaults_when_the_file_is_not_valid(folder):
    (folder / "video_input.json").write_text("{ nope", encoding="utf-8")
    defaults, notes = video_tab.form_defaults(folder)
    assert defaults["duration_s"] == 15 and defaults["voice"] == video_tab.default_voice()
    assert len(notes) == 1 and "not valid JSON" in notes[0]


def test_saving_defaults_keeps_the_keys_the_form_does_not_know(folder):
    path = folder / "video_input.json"
    path.write_text(json.dumps({"note_from_the_team": "keep me", "audience": "old", "duration_s": 15}), encoding="utf-8")

    video_tab.merge_defaults(folder, video_tab.build_overrides({"audience": "families", "duration_s": 10}))

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["note_from_the_team"] == "keep me" and saved["audience"] == "families" and saved["duration_s"] == 10
    assert set(VideoInput.model_fields) <= set(saved)


def test_saving_defaults_creates_the_file_and_refuses_what_would_not_load(folder):
    path = video_tab.merge_defaults(folder, {"audience": "families"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"audience": "families"}
    with pytest.raises(IngestionError, match="not valid"):
        video_tab.merge_defaults(folder, {"duration_s": 7})
    assert json.loads(path.read_text(encoding="utf-8")) == {"audience": "families"}       # untouched
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(IngestionError, match="JSON object"):
        video_tab.merge_defaults(folder, {"audience": "x"})


def test_the_steps_shown_are_the_steps_that_run():
    assert list(video_tab.STEP_TITLES) == workflow._STEP_NAMES


def _summary(**fields) -> VideoSummary:
    now = datetime.now(timezone.utc)
    return VideoSummary(**{"video_id": "vid_x", "property_id": "P", "created_at": now, "status": "planning", **fields})


def test_steps_are_done_current_or_waiting():
    states = [s for _, _, s in video_tab.step_states(_summary(steps_done=3, current_node="script"))]
    assert states == ["done"] * 3 + ["current"] + ["waiting"] * 6
    planned = [s for _, _, s in video_tab.step_states(_summary(status="planned", steps_done=5))]
    assert planned == ["done"] * 5 + ["waiting"] * 5
    failed = [s for _, _, s in video_tab.step_states(_summary(status="failed", steps_done=6, current_node="motion"))]
    assert failed[:6] == ["done"] * 6 and failed[6] == "failed" and set(failed[7:]) == {"waiting"}
    assert [s for _, _, s in video_tab.step_states(_summary(status="passed", steps_done=10))] == ["done"] * 10


def test_elapsed_time_runs_while_the_job_runs_and_stops_when_it_ends():
    start = datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc)
    now = start + timedelta(seconds=75)
    running = _summary(created_at=start - timedelta(hours=1), started_at=start, status="producing")
    assert video_tab.elapsed_seconds(running, now) == 75 and video_tab.clock(75) == "1:15"
    done = _summary(created_at=start, updated_at=start + timedelta(seconds=42), status="passed")
    assert video_tab.elapsed_seconds(done, now) == 42
    assert video_tab.is_active(running) and not video_tab.is_active(done) and not video_tab.is_active(None)


def test_the_log_tail_is_the_last_twenty_lines(tmp_path):
    log = repository.video_output_dir("vid_log0000001") / "work" / "job.log"
    log.parent.mkdir(parents=True)
    log.write_text("\n".join(f"line {n}" for n in range(1, 31)), encoding="utf-8")
    tail = video_tab.log_tail("vid_log0000001").splitlines()
    assert tail[0] == "line 11" and tail[-1] == "line 30" and len(tail) == 20
    assert video_tab.log_tail("vid_nolog000001") == ""


# -- what wins: the toolbar, the form, the prompt, the property's file -------------------------------------------

def _toolbar(**fields) -> video_box.Toolbar:
    return video_box.Toolbar(**{"property_id": PROPERTY, "prompt": "p", **fields})


def test_with_plan_off_the_toolbar_wins_over_the_prompt_and_the_rest_is_left_to_the_file():
    prompt_values = {"duration_s": 30, "aspect": "9:16", "objective": "launch", "audience": "families"}

    assert video_box.plan_off_overrides(prompt_values, _toolbar()) == prompt_values          # Auto: the prompt says
    chosen = video_box.plan_off_overrides(prompt_values, _toolbar(length=45, shape="1:1", model="kenburns"))
    assert chosen == {"duration_s": 45, "aspect": "1:1", "objective": "launch", "audience": "families",
                      "model": "kenburns"}
    assert video_box.plan_off_overrides({}, _toolbar()) == {}                                # nothing set: nothing sent
    assert "voice" not in video_box.plan_off_overrides(prompt_values, _toolbar())


DEFAULTS = {"duration_s": 15, "aspect": "9:16", "audience": "old audience", "objective": "awareness",
            "highlights": "Gym", "tone": "refined", "voice": "default", "music_mood": "", "cta": "",
            "creative_direction": "", "hero_image": "", "burn_captions": True, "grade": True, "allow_generated": False,
            "pronunciations": "", "model": "auto"}


def test_the_prompt_fills_only_the_form_fields_the_person_has_not_changed():
    values = {f: DEFAULTS[f] for f in video_tab.FORM_FIELDS}
    values["tone"] = "warm"                                                    # changed by the person
    prompt_values = {"audience": "young families", "tone": "confident", "objective": "launch",
                     "highlights": ["Swimming pool", "Clubhouse"], "burn_captions": False,
                     "pronunciations": {"Baner": "Bah-ner"}, "cta": "Book a visit"}

    merged = video_box.merge_prompt(values, DEFAULTS, prompt_values)

    assert merged["tone"] == "warm"                                           # the person's value stands
    assert merged["audience"] == "young families" and merged["objective"] == "launch"
    assert merged["highlights"] == "Swimming pool\nClubhouse" and merged["cta"] == "Book a visit"
    assert merged["burn_captions"] is False and merged["pronunciations"] == "Baner=Bah-ner"
    assert merged["voice"] == "default" and merged["grade"] is True            # the prompt said nothing about these


def test_a_field_the_person_cleared_is_not_filled_from_the_prompt():
    values = {f: DEFAULTS[f] for f in video_tab.FORM_FIELDS}
    values["audience"] = ""
    assert video_box.merge_prompt(values, DEFAULTS, {"audience": "families"})["audience"] == ""


def test_with_plan_on_the_length_shape_and_model_are_the_toolbars_then_the_prompts_then_the_files():
    values = {f: DEFAULTS[f] for f in video_tab.FORM_FIELDS}
    assert video_box.with_toolbar(values, DEFAULTS, _toolbar())["duration_s"] == 15
    assert video_box.with_toolbar(values, DEFAULTS, _toolbar(), {"duration_s": 30, "aspect": "1:1"})["duration_s"] == 30
    both = video_box.with_toolbar(values, DEFAULTS, _toolbar(length=10, shape="16:9", model="kenburns"),
                                  {"duration_s": 30, "aspect": "1:1"})
    assert (both["duration_s"], both["aspect"], both["model"]) == (10, "16:9", "kenburns")
    assert video_box.with_toolbar(values, {**DEFAULTS, "model": "kenburns"}, _toolbar())["model"] == "kenburns"


def test_plan_on_sends_every_option_and_they_are_valid():
    values = {f: DEFAULTS[f] for f in video_tab.FORM_FIELDS}
    overrides = video_box.plan_on_overrides(values, DEFAULTS, {"objective": "launch", "duration_s": 30}, _toolbar())
    assert set(overrides) == set(VideoInput.model_fields)
    assert overrides["objective"] == "launch" and overrides["duration_s"] == 30
    VideoInput.model_validate(overrides)


# -- the estimate ------------------------------------------------------------------------------------------------

def test_with_no_key_a_video_is_free_and_the_motion_is_named():
    est = video_box.estimate(15, "auto", offline=False)
    assert (est.usd, est.credits, est.offline) == (0.0, False, False)
    assert video_box.estimate_text(est) == "Estimate: free · 15 s · Pan-and-zoom (free)"


def test_offline_is_free_and_says_nothing_leaves_the_computer():
    est = video_box.estimate(30, "auto", offline=True)
    assert est.usd == 0 and video_box.estimate_text(est).startswith("Estimate: free. Offline")


def test_the_voice_and_the_music_are_priced_from_the_configs_when_their_services_have_keys(monkeypatch):
    monkeypatch.setattr(settings, "google_cloud_tts_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "gemini_api_key", "k", raising=False)
    # 13 s spoken at 2.1 words a second and 6 characters a word, $30 per million characters; one Lyria clip at $0.04
    expected = 13 * 2.1 * 6 * 30 / 1_000_000 + 0.04
    est = video_box.estimate(15, "auto", offline=False)
    assert est.usd == pytest.approx(expected)
    assert video_box.estimate_text(est) == "Estimate: about $0.04 · 15 s · Pan-and-zoom (free)"
    assert video_box.estimate(60, "auto", offline=False).usd == pytest.approx(58 * 2.1 * 6 * 30 / 1_000_000 + 0.08)


def test_a_cost_under_a_cent_is_said_so_and_a_half_cent_rounds_up():
    assert video_box.estimate_text(video_box.Estimate(0.004, False, 15, "Pan-and-zoom (free)", False)) \
        == "Estimate: under $0.01 · 15 s · Pan-and-zoom (free)"
    assert video_box.estimate_text(video_box.Estimate(0.075, False, 15, "Eternal AI Wan 2.2 I2V", False)) \
        == "Estimate: about $0.08 · 15 s · Eternal AI Wan 2.2 I2V"


def test_a_credit_based_model_is_priced_plus_credits(monkeypatch):
    class Adapter(MockVideoProvider):
        pass

    monkeypatch.setitem(registry.ADAPTERS, "magic_hour", Adapter)
    monkeypatch.setenv("MAGIC_HOUR_API_KEY", "k")
    est = video_box.estimate(15, "magic_hour_wan22_i2v", offline=False)
    assert est.credits is True and est.motion == "Magic Hour Wan 2.2 I2V"
    assert video_box.estimate_text(est) == "Estimate: credits only · 15 s · Magic Hour Wan 2.2 I2V"
    paid = video_box.Estimate(0.04, True, 15, "Magic Hour Wan 2.2 I2V", False)
    assert video_box.estimate_text(paid) == "Estimate: about $0.04 plus credits · 15 s · Magic Hour Wan 2.2 I2V"


def test_a_dollar_sign_in_a_caption_is_escaped_so_two_are_not_read_as_maths():
    assert video_tab.md("from about $0.08 · $2 signup credits") == "from about \\$0.08 · \\$2 signup credits"


# -- the page: the box ----------------------------------------------------------------------------------------------

def _choose(at, prompt="A calm reel for families", property_id=PROPERTY):
    at.radio(key="vt_property").set_value(property_id)
    at.text_area(key="vt_prompt").set_value(prompt)
    return at.run()


def _estimate_line(at) -> str:
    return next(str(c.value).replace("\\$", "$") for c in at.caption if str(c.value).startswith("Estimate"))


def test_the_video_tab_opens_on_the_empty_box(folder):
    at = _app().run()

    assert not at.exception
    assert at.radio(key="solution").options == ["Brochure", "Blog post", "Video"]
    assert not any("Project name" in w.label for w in at.text_input)               # not the brochure form
    prompt = at.text_area(key="vt_prompt")
    assert prompt.value == "" and prompt.max_chars == 4000
    for said in ("who it is for", "what to highlight", "the mood", "the call to action",
                 "Camera moves and on-screen text are decided by the tool"):
        assert said in prompt.placeholder
    assert at.radio(key="vt_property").options == ["Aurelia Heights (AURELIA-HEIGHTS)"]
    assert at.radio(key="vt_property").value is None
    assert at.radio(key="vt_shape").options == ["Auto", "9:16", "16:9", "1:1"] and at.radio(key="vt_shape").value == "auto"
    assert at.radio(key="vt_length").options == ["Auto", "10 s", "15 s", "30 s", "45 s", "60 s"]
    assert at.radio(key="vt_length").value == "auto"
    assert at.checkbox(key="vt_offline").value is False and at.checkbox(key="vt_draft").value is False
    assert at.toggle(key="vt_plan").value is False                                 # Plan starts off
    recipes = at.button(key="vt_recipes")
    assert recipes.label == "Recipes, coming soon" and recipes.disabled
    assert at.button(key="vt_send").disabled
    assert "Choose a property with + before sending." in _shown(at)
    assert _estimate_line(at) == "Estimate: free · 15 s · Pan-and-zoom (free)"
    labels = [w.label for w in (*at.selectbox, *at.text_input)]
    assert "Who is it for?" not in labels and "Voice" not in labels                  # no form while Plan is off
    assert not [b for b in at.button if b.key in ("vt_start", "vt_save")]


def test_the_toolbar_buttons_show_the_current_values(folder):
    at = _app().run()
    labels = {p.proto.popover.label for p in at.get("popover")}
    assert labels == {"+ Property", "Model: Auto", "Shape: Auto", "More"}

    at.radio(key="vt_shape").set_value("16:9")
    at.radio(key="vt_length").set_value(30)
    at.checkbox(key="vt_offline").check()
    at.checkbox(key="vt_draft").check()
    _choose(at, property_id=PROPERTY)
    labels = {p.proto.popover.label for p in at.get("popover")}
    assert labels == {"+ Aurelia Heights", "Model: Auto", "Shape: 16:9", "More · 30 s · offline · draft"}


def test_only_folders_with_a_property_json_are_offered_and_they_are_named(folder, tmp_path):
    (folder.parent / "NOT-A-PROPERTY").mkdir()
    (folder.parent / "stray.txt").write_text("x", encoding="utf-8")
    assert video_tab.property_choices() == [(PROPERTY, "Aurelia Heights (AURELIA-HEIGHTS)")]
    assert _app().run().radio(key="vt_property").options == ["Aurelia Heights (AURELIA-HEIGHTS)"]


def test_with_no_property_the_page_says_what_to_do(tmp_path):
    at = _app().run()
    assert not at.exception and "no property to make a video of" in _shown(at)
    assert not at.get("popover")


def test_the_photographs_of_the_chosen_property_are_shown_above_the_prompt(folder):
    at = _choose(_app().run())
    assert not at.exception
    assert any("photographs" in str(c.value) for c in at.caption)
    strips = [str(m.value) for m in at.markdown if 'class="vt-strip"' in str(m.value)]
    assert len(strips) == 1 and strips[0].count("<img ") >= 3                       # one row of pictures, as data addresses
    assert "overflow-x: auto" in " ".join(str(m.value) for m in at.markdown)         # that scrolls sideways


@pytest.mark.parametrize("property_id,prompt,disabled", [
    (None, "", True), (None, "A calm reel", True), (PROPERTY, "", True), (PROPERTY, "   \n ", True),
    (PROPERTY, "A calm reel", False)])
def test_send_waits_for_a_property_and_a_prompt(property_id, prompt, disabled, folder):
    at = _app().run()
    if property_id:
        at.radio(key="vt_property").set_value(property_id)
    at.text_area(key="vt_prompt").set_value(prompt)
    at.run()
    assert at.button(key="vt_send").disabled is disabled


def test_the_estimate_follows_the_length_the_prompt_or_the_toolbar_names(folder):
    at = _choose(_app().run(), "A 30 second reel")
    assert _estimate_line(at) == "Estimate: free · 30 s · Pan-and-zoom (free)"
    assert "have not been rendered" in _shown(at)
    at.radio(key="vt_length").set_value(10).run()                                    # the toolbar wins over the prompt
    assert _estimate_line(at) == "Estimate: free · 10 s · Pan-and-zoom (free)"
    assert "have not been rendered" not in _shown(at)
    at.checkbox(key="vt_offline").check().run()
    assert _estimate_line(at).startswith("Estimate: free. Offline")


def test_a_property_that_names_its_own_length_sets_the_estimate_until_the_toolbar_says_otherwise(folder):
    (folder / "video_input.json").write_text(json.dumps({"duration_s": 45}), encoding="utf-8")
    at = _choose(_app().run())
    assert _estimate_line(at) == "Estimate: free · 45 s · Pan-and-zoom (free)"


def _rows(at) -> dict[str, str]:
    """The "not available yet" rows of the model list, by the label each starts with (the escapes of a dollar
    sign undone)."""
    rows = [str(c.value).replace("\\$", "$") for c in at.caption if " · not available yet" in str(c.value)]
    return {row.split(" · ")[0]: row for row in rows}


def test_the_model_menu_offers_what_can_run_and_lists_the_rest_as_not_available_yet(folder):
    at = _app().run()

    radio = at.radio(key="vt_model")
    assert radio.options[0].startswith("Auto") and radio.value == "auto"
    assert len(radio.options) == 2 and radio.options[1].startswith("Pan-and-zoom") and "free" in radio.options[1]
    assert "**Not available yet**" in [str(m.value) for m in at.markdown]
    rows = _rows(at)
    assert list(rows) == ["MiniMax H3 Max", "Kling 2.5 Turbo Pro", "Seedance 2.0", "Magic Hour Wan 2.2 I2V",
                          "8Scale Wan 2.2 14B I2V", "Nodaro Seedance 2 Fast", "Nodaro Veo 3.1 Lite",
                          "Eternal AI Wan 2.2 I2V"]
    assert "via fal.ai" in rows["MiniMax H3 Max"] and "about $0.45" in rows["MiniMax H3 Max"]       # 0.03 a second, 15 s
    assert "price not set" in rows["Seedance 2.0"]
    assert at.radio(key="vt_model").disabled is False
    at.checkbox(key="vt_offline").check().run()
    assert at.radio(key="vt_model").disabled is True


def test_a_row_names_the_estimate_the_free_tier_and_what_to_watch_for(folder):
    rows = _rows(_app().run())

    assert rows["Magic Hour Wan 2.2 I2V"] == ("Magic Hour Wan 2.2 I2V · not available yet · credit-based · "
                                              "free signup credits · price unverified")
    assert rows["Nodaro Seedance 2 Fast"] == ("Nodaro Seedance 2 Fast · not available yet · credit-based · "
                                              "free tier up to 4 s at the lowest resolution · "
                                              "free tier watermarked · price unverified")
    assert rows["8Scale Wan 2.2 14B I2V"] == ("8Scale Wan 2.2 14B I2V · not available yet · from about $0.05 · "
                                              "10 free generations · price unverified")
    assert rows["Eternal AI Wan 2.2 I2V"] == ("Eternal AI Wan 2.2 I2V · not available yet · from about $0.08 · "
                                              "$2 signup credits · price unverified")


def test_a_longer_video_is_estimated_longer(folder):
    at = _app().run()
    at.radio(key="vt_length").set_value(60).run()
    rows = _rows(at)
    assert "about $1.80" in rows["MiniMax H3 Max"]
    assert "from about $0.20" in rows["8Scale Wan 2.2 14B I2V"] and "from about $0.30" in rows["Eternal AI Wan 2.2 I2V"]


def test_a_model_that_is_no_longer_offered_is_dropped_from_the_toolbar(folder):
    at = _app()
    at.session_state["vt_model"] = "seedance_2"
    at.session_state["vt_shape"] = "21:9"
    at.run()
    assert not at.exception and at.radio(key="vt_model").value == "auto" and at.radio(key="vt_shape").value == "auto"


# -- the page: send with Plan off ----------------------------------------------------------------------------------

@pytest.fixture
def read(monkeypatch):
    """prompt_brief.read_prompt as a stub: it records what it was asked and answers with `.result`."""
    seen = SimpleNamespace(calls=[], result=PromptBriefResult(PromptBrief(), [], "rules"), raises=None)

    def fake(prompt, property_dir, offline=False, **kwargs):
        seen.calls.append((prompt, Path(property_dir).name, offline))
        if seen.raises:
            raise seen.raises
        return seen.result

    monkeypatch.setattr(prompt_brief, "read_prompt", fake)
    return seen


def test_plan_off_reads_the_prompt_and_starts_the_full_job_with_the_toolbar_on_top(folder, calls, read):
    read.result = PromptBriefResult(
        PromptBrief(duration_s=30, aspect="9:16", objective="launch", audience="families",
                    highlights=["Swimming pool"]),
        ["'helipad' is not in this property's details; left out."], "model")
    at = _app().run()
    at.radio(key="vt_shape").set_value("1:1")
    at.radio(key="vt_length").set_value(45)
    at.radio(key="vt_model").set_value("kenburns")
    at = _choose(at, "A 30 second reel to launch it for families")
    at.button(key="vt_send").click().run()

    assert not at.exception
    assert read.calls == [("A 30 second reel to launch it for families", PROPERTY, False)]
    (property_id, overrides, flags), = calls["plan"]
    assert property_id == PROPERTY
    assert overrides == {"duration_s": 45, "aspect": "1:1", "objective": "launch", "audience": "families",
                         "highlights": ["Swimming pool"], "model": "kenburns"}          # the toolbar's, over the prompt's
    assert flags == {"offline": False, "draft": False, "auto_approve": True,
                     "prompt": "A 30 second reel to launch it for families", "brief_source": "model",
                     "brief_notes": ["'helipad' is not in this property's details; left out."],
                     "sources": {"duration_s": "toolbar", "aspect": "toolbar", "objective": "prompt",
                                 "audience": "prompt", "tone": "default", "highlights": "prompt", "cta": "default",
                                 "music_mood": "default", "voice": "default", "model": "toolbar"}}
    assert at.query_params["video"] in ("vid_stub000001", ["vid_stub000001"])


def test_with_auto_everywhere_the_prompt_decides_and_the_voice_is_never_sent(folder, calls, read):
    read.result = PromptBriefResult(PromptBrief(duration_s=30, aspect="16:9", tone="warm"), [], "rules")
    at = _choose(_app().run(), "A 30 second landscape film")
    at.button(key="vt_send").click().run()

    (_, overrides, flags), = calls["plan"]
    assert overrides == {"duration_s": 30, "aspect": "16:9", "tone": "warm"}            # no model, no voice
    assert flags["auto_approve"] is True and flags["brief_source"] == "rules"


def test_an_empty_prompt_result_sends_no_options_at_all(folder, calls, read):
    at = _choose(_app().run())
    at.button(key="vt_send").click().run()
    assert calls["plan"][0][1] == {}


def test_offline_and_draft_are_passed_on_and_the_prompt_is_read_by_rules_alone(folder, calls, read):
    at = _app().run()
    at.checkbox(key="vt_offline").check()
    at.checkbox(key="vt_draft").check()
    at = _choose(at)
    at.button(key="vt_send").click().run()

    assert read.calls[0][2] is True
    (_, _, flags), = calls["plan"]
    assert flags["offline"] is True and flags["draft"] is True


def test_a_property_that_names_its_model_keeps_it_when_the_toolbar_says_auto(folder, calls, read):
    (folder / "video_input.json").write_text(json.dumps({"model": "kenburns"}), encoding="utf-8")
    at = _choose(_app().run())
    at.button(key="vt_send").click().run()
    assert "model" not in calls["plan"][0][1]                                           # video_input.json decides


def test_a_prompt_that_cannot_be_read_says_so_and_starts_nothing(folder, calls, read):
    read.raises = IngestionError("Missing property.json in somewhere.")
    at = _choose(_app().run())
    at.button(key="vt_send").click().run()
    assert "Missing property.json" in " ".join(e.value for e in at.error) and calls["plan"] == []

    read.raises = RuntimeError("boom")
    at.button(key="vt_send").click().run()
    assert "The prompt could not be read (RuntimeError)" in " ".join(e.value for e in at.error)
    assert calls["plan"] == [] and not at.exception


def test_a_busy_job_is_named_with_its_step_and_can_be_opened(folder, monkeypatch, read):
    _seed("vid_busy0000001", "planning", current_node="script", steps_done=3)
    repository.set_pid("vid_busy0000001", os.getpid())

    def busy(*args, **kwargs):
        raise JobBusyError("Video vid_busy0000001 is still planning.")

    monkeypatch.setattr(video_jobs, "start_plan", busy)
    at = _choose(_app().run())
    at.button(key="vt_send").click().run()

    warning = " ".join(w.value for w in at.warning)
    assert "vid_busy0000001" in warning and "write the script" in warning and "planning" in warning
    assert "Only one video is made at a time" in warning
    at.button(key="vt_open_vid_busy0000001").click().run()
    assert at.query_params["video"] in ("vid_busy0000001", ["vid_busy0000001"])
    assert "Planning the video" in _shown(at)


def test_a_refused_start_is_explained(folder, monkeypatch, read):
    def refuse(*args, **kwargs):
        raise JobError("The video options are not valid: duration_s: x.")

    monkeypatch.setattr(video_jobs, "start_plan", refuse)
    at = _choose(_app().run())
    at.button(key="vt_send").click().run()
    assert "The video options are not valid" in " ".join(e.value for e in at.error)


# -- the page: Plan on ------------------------------------------------------------------------------------------

def _plan_on(at, **toolbar):
    at.toggle(key="vt_plan").set_value(True)
    return at.run()


def test_plan_on_shows_the_form_at_once_filled_from_the_property_file(folder):
    (folder / "video_input.json").write_text(json.dumps({
        "audience": "young families", "objective": "launch", "highlights": ["Swimming Pool", "Gym"], "tone": "warm",
        "burn_captions": False, "pronunciations": {"Anamika": "Ah-nah-mih-kah"}, "voice": "male"}), encoding="utf-8")
    at = _plan_on(_app().run())
    assert "Choose a property to see the form." in _shown(at)                         # the form needs a property

    at.radio(key="vt_property").set_value(PROPERTY)
    at.run()
    assert not at.exception
    labels = [w.label for w in (*at.selectbox, *at.text_input, *at.text_area, *at.checkbox)]
    for expected in ("What it is for", "Who is it for?", "Key highlights to feature (one per line)",
                     "Call to action", "Creative direction", "Tone", "Music mood", "Voice", "Opening picture",
                     "Print the narration on screen", "Light colour grade", "Allow generated frames",
                     "Pronunciations (one per line: word=how it is said)"):
        assert expected in labels, expected
    for covered in ("Length", "Shape", "Model"):                                     # the toolbar has these
        assert covered not in labels
    assert at.selectbox(key=_key("objective")).options == ["Awareness", "Site visit", "Launch", "Enquiry"]
    assert at.selectbox(key=_key("objective")).value == "launch" and at.selectbox(key=_key("tone")).value == "warm"
    assert at.selectbox(key=_key("voice")).value == "male"
    assert at.text_input(key=_key("audience")).value == "young families"
    assert at.text_area(key=_key("highlights")).value == "Swimming Pool\nGym"
    assert at.text_area(key=_key("pronunciations")).value == "Anamika=Ah-nah-mih-kah"
    assert at.checkbox(key=_key("burn_captions")).value is False and at.checkbox(key=_key("grade")).value is True
    assert at.checkbox(key=_key("allow_generated")).value is False
    assert at.button(key="vt_save") is not None


def test_plan_on_takes_the_prompts_value_for_untouched_fields_and_keeps_the_persons_edits(folder, calls, read):
    (folder / "video_input.json").write_text(json.dumps({"audience": "old audience", "tone": "refined"}),
                                             encoding="utf-8")
    read.result = PromptBriefResult(
        PromptBrief(duration_s=30, audience="young families", tone="confident", objective="launch",
                    highlights=["Swimming pool"], cta="Book a visit"), ["a note"], "model")
    at = _app().run()
    at.radio(key="vt_property").set_value(PROPERTY)
    at = _plan_on(at)
    at.selectbox(key=_key("tone")).select("warm")                                     # the person's own edit
    at.text_area(key=_key("creative_direction")).set_value("Slow and calm")           # and another
    at.text_area(key="vt_prompt").set_value("A 30 second reel to launch it")
    at.run()
    at.button(key="vt_send").click().run()

    assert not at.exception
    (property_id, overrides, flags), = calls["plan"]
    assert set(overrides) == set(VideoInput.model_fields)                              # the form sends every option
    assert overrides["audience"] == "young families"                                   # untouched: the prompt's
    assert overrides["objective"] == "launch" and overrides["highlights"] == ["Swimming pool"]
    assert overrides["cta"] == "Book a visit"
    assert overrides["tone"] == "warm" and overrides["creative_direction"] == "Slow and calm"   # changed: the person's
    assert overrides["duration_s"] == 30 and overrides["aspect"] == "9:16" and overrides["model"] == "auto"
    assert flags["auto_approve"] is False and flags["brief_source"] == "model" and flags["brief_notes"] == ["a note"]
    assert flags["sources"] == {"duration_s": "prompt", "aspect": "default", "objective": "prompt", "audience": "prompt",
                                "tone": "form", "highlights": "prompt", "cta": "prompt", "music_mood": "default",
                                "voice": "default", "model": "default"}
    assert flags["prompt"] == "A 30 second reel to launch it"
    assert at.query_params["video"] in ("vid_stub000001", ["vid_stub000001"])


def test_plan_on_puts_the_toolbar_over_the_prompt_and_the_form(folder, calls, read):
    read.result = PromptBriefResult(PromptBrief(duration_s=30, aspect="16:9"), [], "rules")
    at = _app().run()
    at.radio(key="vt_property").set_value(PROPERTY)
    at.radio(key="vt_length").set_value(10)
    at.radio(key="vt_shape").set_value("1:1")
    at.radio(key="vt_model").set_value("kenburns")
    at = _plan_on(at)
    at.text_area(key="vt_prompt").set_value("A video")
    at.run()
    at.button(key="vt_send").click().run()

    (_, overrides, _), = calls["plan"]
    assert (overrides["duration_s"], overrides["aspect"], overrides["model"]) == (10, "1:1", "kenburns")


def test_plan_on_starts_the_plan_job_and_not_the_full_one(folder, calls, read):
    at = _app().run()
    at.radio(key="vt_property").set_value(PROPERTY)
    at = _plan_on(at)
    at.text_area(key="vt_prompt").set_value("A video")
    at.run()
    at.button(key="vt_send").click().run()
    assert calls["plan"][0][2]["auto_approve"] is False


def test_save_as_defaults_writes_the_form_and_the_toolbar_choices(folder, read):
    at = _app().run()
    at.radio(key="vt_property").set_value(PROPERTY)
    at.radio(key="vt_length").set_value(30)
    at = _plan_on(at)
    at.text_input(key=_key("audience")).set_value("retirees")
    at.button(key="vt_save").click().run()

    saved = json.loads((folder / "video_input.json").read_text(encoding="utf-8"))
    assert saved["audience"] == "retirees" and saved["duration_s"] == 30
    assert "Saved to video_input.json" in _shown(at)
    assert read.calls == []                                                            # saving reads no prompt


def test_the_form_starts_from_the_defaults_when_the_file_is_not_valid_and_says_so(folder):
    (folder / "video_input.json").write_text("{not json", encoding="utf-8")
    at = _app().run()
    at.radio(key="vt_property").set_value(PROPERTY)
    at = _plan_on(at)
    assert not at.exception and "is not valid JSON" in " ".join(w.value for w in at.warning)
    assert at.selectbox(key=_key("objective")).value == "awareness"


# -- a video: progress, approval, result -------------------------------------------------------------------

def test_a_link_to_a_video_opens_the_video_tab(folder):
    _seed()
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["video"] = "vid_seed000001"
    at.run()
    assert at.radio(key="solution").value == "Video"


def test_progress_shows_the_ten_steps_and_a_cancel_button(folder, calls):
    _seed(current_node="script", steps_done=3)
    repository.update_progress("vid_seed000001", current_node="script",
                               steps=[AgentStepResult(node=n, latency_ms=1) for n in workflow._STEP_NAMES[:3]])
    at = _app(video="vid_seed000001").run()

    assert not at.exception
    text = _shown(at)
    assert "Planning the video" in text and "elapsed" in text
    for step, title in video_tab.STEP_TITLES.items():
        assert title in text, step
    assert text.count("✅") == 3 and text.count("⏳") == 1 and text.count("▫️") == 6
    at.button(key="vt_cancel_vid_seed000001").click().run()
    assert calls["cancel"] == ["vid_seed000001"]


def test_progress_polling_reads_the_database_only(folder, monkeypatch):
    _seed()
    monkeypatch.setattr(repository, "get_generation", lambda *a, **k: pytest.fail("the record was parsed"))
    summary = video_tab.render_progress("vid_seed000001")
    assert summary.status == "planning"
    assert video_tab.render_progress("vid_unknown0001") is None


def test_a_job_whose_process_died_is_shown_as_failed_with_its_log(folder, monkeypatch):
    _seed(current_node="script")
    repository.set_pid("vid_seed000001", 999_999)
    log = repository.video_output_dir("vid_seed000001") / "work" / "job.log"
    log.parent.mkdir(parents=True)
    log.write_text("\n".join(f"log line {n}" for n in range(1, 41)), encoding="utf-8")
    monkeypatch.setattr(video_jobs, "pid_alive", lambda pid: False)

    at = _app(video="vid_seed000001").run()

    assert not at.exception
    assert "couldn't finish this video" in " ".join(e.value for e in at.error)
    assert "interrupted" in " ".join(e.value for e in at.error)
    code = " ".join(c.value for c in at.code)
    assert "log line 40" in code and "log line 21" in code and "log line 20" not in code


def test_a_cancelled_video_says_so(folder):
    _seed(status="cancelled")
    at = _app(video="vid_seed000001").run()
    assert "was cancelled" in " ".join(w.value for w in at.warning)


def test_an_unknown_video_is_said_plainly(folder):
    at = _app(video="vid_nothere0001").run()
    assert not at.exception and "There is no video vid_nothere0001" in " ".join(w.value for w in at.warning)


def _planned(folder) -> VideoGenerationRecord:
    return workflow.run_plan_workflow(folder, {"audience": "families"}, offline=True, draft=True)


def test_the_approval_view_shows_the_script_the_shots_and_the_notes(folder, calls):
    record = _planned(folder)
    at = _app(video=record.video_id).run()

    assert not at.exception
    text = _shown(at)
    assert "Review the plan" in text and "Aurelia Heights" in text
    for line in record.script.lines:
        assert line.text in text
    assert f"{record.script.word_count} words; the plan aimed for about {record.brief.target_words}" in text
    for shot in record.storyboard.shots:
        assert f"Shot {shot.index + 1}" in text
    assert "fills the frame" in text or "whole photograph" in text
    assert "Closing card" in text and "On screen:" in text
    assert [b.label for b in at.button if b.key and b.key.startswith("vt_")][-2:] == ["Approve and produce", "Re-plan"]
    assert not any("Notes about this plan" in e.label for e in at.expander)


def test_approving_starts_the_produce_job(folder, calls):
    record = _planned(folder)
    at = _app(video=record.video_id).run()
    at.button(key=f"vt_approve_{record.video_id}").click().run()
    assert calls["produce"] == [record.video_id] and not at.exception


def test_approving_while_another_video_runs_says_which(folder, monkeypatch):
    record = _planned(folder)
    _seed("vid_busy0000001", "producing", current_node="motion", steps_done=6)
    repository.set_pid("vid_busy0000001", os.getpid())

    def busy(video_id):
        raise JobBusyError("busy")

    monkeypatch.setattr(video_jobs, "start_produce", busy)
    at = _app(video=record.video_id).run()
    at.button(key=f"vt_approve_{record.video_id}").click().run()
    assert "vid_busy0000001" in " ".join(w.value for w in at.warning)


def test_re_plan_starts_a_new_plan_from_the_stored_request_and_keeps_the_old_one(folder, calls):
    record = _planned(folder)
    at = _app(video=record.video_id).run()
    at.button(key=f"vt_replan_{record.video_id}").click().run()

    (property_id, overrides, flags), = calls["plan"]
    assert property_id == record.request.property_id == PROPERTY
    assert overrides == {"audience": "families"}
    assert flags == {"offline": True, "draft": True, "refresh_ledger": False, "prompt": "", "brief_source": "",
                     "brief_notes": [], "sources": {}}
    assert at.query_params["video"] in ("vid_stub000001", ["vid_stub000001"])
    old = repository.get_generation(record.video_id)
    assert old.status == "planned" and old.script == record.script and (Path(old.output_dir) / PLAN_STATE_FILE).is_file()


def _finished(status, tmp_path, with_video=True, gate=VideoGate.PASS, score=100.0) -> VideoGenerationRecord:
    repository.init_db()
    out = repository.video_output_dir("vid_done0000001")
    out.mkdir(parents=True, exist_ok=True)
    video = out / "video.mp4"
    if with_video:
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 2000)
    record = VideoGenerationRecord(
        video_id="vid_done0000001", property_id=PROPERTY, status=status, video_path=str(video) if with_video else None,
        evaluation=VideoEvaluationResult(quality_score=score, gate_result=gate),
        error=None if status != "failed" else "The quality gate failed: duration")
    repository.save_generation(record)
    return record


@pytest.mark.parametrize("status,gate,shown", [("passed", VideoGate.PASS, "The video is ready."),
                                              ("warned", VideoGate.WARN, "with a few things worth a look")])
def test_a_finished_video_is_played_with_its_score(status, gate, shown, folder, tmp_path):
    _finished(status, tmp_path, gate=gate, score=91.0)
    at = _app(video="vid_done0000001").run()

    assert not at.exception
    text = " ".join([_shown(at)])
    assert shown in text and f"{gate.value.upper()} — score **91/100**" in text
    assert "Evaluation and history come next." in text
    assert len(at.get("video")) == 1


def test_a_video_that_failed_the_gate_is_still_shown_with_the_verdict(folder, tmp_path):
    _finished("failed", tmp_path, gate=VideoGate.FAIL, score=40.0)
    at = _app(video="vid_done0000001").run()
    assert "did not pass the quality check" in " ".join(e.value for e in at.error)
    assert len(at.get("video")) == 1 and "FAIL" in _shown(at)


def test_a_failure_with_no_video_is_plain_and_shows_the_log(folder, tmp_path):
    _finished("failed", tmp_path, with_video=False)
    log = repository.video_output_dir("vid_done0000001") / "work" / "job.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("Traceback (most recent call last)\nRuntimeError: boom", encoding="utf-8")
    at = _app(video="vid_done0000001").run()

    assert not at.exception and len(at.get("video")) == 0
    assert "couldn't finish this video" in " ".join(e.value for e in at.error)
    assert "RuntimeError: boom" in " ".join(c.value for c in at.code)


def test_re_plan_keeps_the_prompt_and_what_was_made_of_it(folder, calls):
    record = _planned(folder)
    record.request = record.request.model_copy(update={"prompt": "A calm reel", "brief_source": "model",
                                                       "brief_notes": ["a note"], "sources": {"tone": "form"}})
    repository.save_generation(record)
    at = _app(video=record.video_id).run()
    at.button(key=f"vt_replan_{record.video_id}").click().run()

    (_, _, flags), = calls["plan"]
    assert (flags["prompt"], flags["brief_source"], flags["brief_notes"]) == ("A calm reel", "model", ["a note"])
    assert flags["sources"] == {"tone": "form"}
    assert flags.get("auto_approve") in (None, False)


# -- "Your prompt" on the views of a video made from a prompt -------------------------------------------------------

PROMPT_NOTES = ["'helipad' is not in this property's details; left out.", "Length 20 s is not offered; using 15 s."]


def _from_prompt(record: VideoGenerationRecord, source="model") -> VideoGenerationRecord:
    request = (record.request or VideoRequest(property_id=PROPERTY)).model_copy(update={
        "prompt": "A calm 20 second reel. Highlight the helipad.", "brief_source": source, "brief_notes": PROMPT_NOTES})
    record.request = request
    repository.save_generation(record)
    return record


def _prompt_panel(at):
    return next((e for e in at.expander if e.label == "Your prompt"), None)


def test_the_progress_view_shows_the_prompt_how_it_was_read_and_the_notes(folder, calls):
    _from_prompt(_seed("vid_prompt00001", "planning", current_node="script", steps_done=3))
    repository.set_pid("vid_prompt00001", os.getpid())
    at = _app(video="vid_prompt00001").run()

    assert not at.exception
    panel = _prompt_panel(at)
    assert panel is not None
    assert [t.value for t in panel.text] == ["A calm 20 second reel. Highlight the helipad."]
    shown = " ".join(str(e.value) for e in (*panel.markdown, *panel.caption))
    assert "Read by the model, then checked against the property's facts." in shown
    for note in PROMPT_NOTES:
        assert note in shown
    assert "Planning the video" in _shown(at)                                          # the progress view as before


def test_the_approval_and_result_views_show_it_too(folder, calls, tmp_path):
    planned = _from_prompt(_planned(folder), source="rules")
    at = _app(video=planned.video_id).run()
    panel = _prompt_panel(at)
    assert panel is not None and "Read by built-in rules (the model was not used)." in " ".join(
        str(c.value) for c in panel.caption)
    assert "Review the plan" in _shown(at)

    done = _from_prompt(_finished("passed", tmp_path))
    at = _app(video=done.video_id).run()
    assert _prompt_panel(at) is not None and "The video is ready." in _shown(at)


def test_a_video_made_from_the_form_has_no_prompt_panel(folder, calls):
    planned = _planned(folder)
    at = _app(video=planned.video_id).run()
    assert _prompt_panel(at) is None and "Review the plan" in _shown(at)


# -- where each option came from ---------------------------------------------------------------------------------

def test_a_value_comes_from_the_first_of_the_toolbar_the_form_the_prompt_the_saved_file_or_the_default():
    toolbar = video_box.Toolbar(PROPERTY, "p", length=10)
    sources = video_box.option_sources(toolbar, form_changed={"tone", "audience"},
                                       prompt_keys={"duration_s", "audience", "objective", "tone", "cta"},
                                       saved={"tone", "highlights", "objective", "voice"})
    assert sources == {"duration_s": "toolbar", "aspect": "default", "objective": "prompt", "audience": "form",
                       "tone": "form", "highlights": "saved defaults", "cta": "prompt", "music_mood": "default",
                       "voice": "saved defaults", "model": "default"}


def test_a_chosen_model_or_offline_is_the_toolbars():
    assert video_box.option_sources(video_box.Toolbar(PROPERTY, "p", model="kenburns"), set(), set(), set())["model"] \
        == "toolbar"
    assert video_box.option_sources(video_box.Toolbar(PROPERTY, "p", offline=True), set(), set(), set())["model"] \
        == "toolbar"
    assert video_box.option_sources(video_box.Toolbar(PROPERTY, "p"), set(), set(), {"model"})["model"] == "saved defaults"


def test_the_saved_keys_are_those_of_the_property_file(folder):
    assert video_box.saved_keys(folder) == set()
    (folder / "video_input.json").write_text(json.dumps({"tone": "warm", "voice": "male"}), encoding="utf-8")
    assert video_box.saved_keys(folder) == {"tone", "voice"}
    (folder / "video_input.json").write_text("{not json", encoding="utf-8")
    assert video_box.saved_keys(folder) == set()


def test_plan_off_keeps_where_each_option_came_from(folder, calls, read):
    (folder / "video_input.json").write_text(json.dumps({"voice": "male", "tone": "warm"}), encoding="utf-8")
    read.result = PromptBriefResult(PromptBrief(tone="refined", objective="launch"), [], "rules")
    at = _app().run()
    at.radio(key="vt_length").set_value(30)
    at = _choose(at)
    at.button(key="vt_send").click().run()

    sources = calls["plan"][0][2]["sources"]
    assert sources["duration_s"] == "toolbar" and sources["tone"] == "prompt" and sources["objective"] == "prompt"
    assert sources["voice"] == "saved defaults" and sources["aspect"] == "default" and sources["cta"] == "default"


def test_the_brief_table_says_where_each_value_came_from(folder, calls):
    record = _planned(folder)
    request = record.request.model_copy(update={
        "prompt": "A calm reel", "brief_source": "rules",
        "sources": {"duration_s": "toolbar", "aspect": "default", "objective": "prompt", "audience": "form",
                    "tone": "saved defaults", "highlights": "prompt", "cta": "default", "music_mood": "default",
                    "voice": "default", "model": "toolbar"}})
    record.request = request
    repository.save_generation(record)
    at = _app(video=record.video_id).run()

    (table,) = at.table
    rows = table.value.to_dict("records")
    assert [r["Option"] for r in rows] == ["Length", "Shape", "Objective", "Audience", "Tone", "Highlights",
                                           "Call to action", "Music mood", "Voice", "Model"]
    by = {r["Option"]: r for r in rows}
    assert by["Length"]["Value"] == "15 seconds" and by["Length"]["From"] == "Toolbar"
    assert by["Audience"]["Value"] == "families" and by["Audience"]["From"] == "Form (changed by you)"
    assert by["Objective"]["From"] == "Your prompt" and by["Tone"]["From"] == "Saved defaults (video_input.json)"
    assert by["Shape"]["From"] == "Built-in default" and by["Model"]["Value"] in ("Auto", "Pan-and-zoom (free)")
    assert by["Voice"]["Value"].startswith("Professional, Indian English")


def test_a_video_made_before_sources_were_kept_says_so_in_the_table(folder, calls):
    record = _planned(folder)
    record.request = record.request.model_copy(update={"prompt": "A calm reel", "brief_source": "model"})
    repository.save_generation(record)
    rows = _app(video=record.video_id).run().table[0].value.to_dict("records")
    assert {r["From"] for r in rows} == {video_tab.NOT_RECORDED}


# -- the review page: about this plan, technical, a short script --------------------------------------------------

TECHNICAL = ["Fact ledger from the cache (made 2026-11-05 by gemini:gemini-3.6-flash): the same property, documents, "
             "model and prompt as then. --refresh-ledger asks the model again.",
             "16 pictures read by a vision model (Vision: 16 from the cache.).",
             "13 usable photographs; set aside: map_location.webp (a map, not a photograph).",
             "Blocked by the run's policy: role 'x' may not be asked in this run."]
ABOUT = ["Left out of the script: 'Wellness Garden and Pet Park'. No photograph shows it.",
         "The script is 34 words against a target of 59: add highlights or more property details for a fuller script.",
         "No RERA number is given for this property, so the closing card shows none.",
         "The shot for 'Outdoor Fitness Area, all on site.' would have shown a.webp again: no other photograph shows "
         "court, so it is joined to the shot before, which is longer."]


def _with_notes(folder, notes, target_words=None) -> VideoGenerationRecord:
    record = _planned(folder)
    record.notes = notes
    if target_words:
        record.brief.target_words = target_words
    repository.save_generation(record)
    return record


def test_the_plans_notes_are_split_into_about_this_plan_and_technical(folder):
    about, technical = video_tab.split_notes(SimpleNamespace(notes=[*TECHNICAL[:2], *ABOUT, *TECHNICAL[2:]],
                                                             defects=[ABOUT[0], "A defect."]))
    assert about == [*ABOUT, "A defect."] and technical == TECHNICAL


def test_the_review_page_shows_about_this_plan_first_and_folds_the_technical_notes(folder):
    record = _with_notes(folder, [*TECHNICAL, *ABOUT])
    at = _app(video=record.video_id).run()

    assert not at.exception
    assert "**About this plan**" in [str(m.value) for m in at.markdown]
    written = [str(m.value) for m in at.markdown]
    for note in ABOUT:
        assert f"- {note}" in written
    (technical,) = [e for e in at.expander if e.label.startswith("Technical")]
    assert technical.label == "Technical (4)" and technical.proto.expanded is False
    assert [str(m.value) for m in technical.markdown] == [f"- {n}" for n in TECHNICAL]
    assert all(written.count(f"- {n}") == 1 for n in TECHNICAL)                  # once, and that is inside the fold
    assert not any(f"- {n}" in [str(m.value) for m in technical.markdown] for n in ABOUT)


def test_a_plan_with_no_technical_notes_has_no_technical_section(folder):
    record = _with_notes(folder, ABOUT)
    assert not [e for e in _app(video=record.video_id).run().expander if e.label.startswith("Technical")]


@pytest.mark.parametrize("words,target,short", [(34, 59, True), (42, 59, False), (40, 59, True), (0, 59, True),
                                               (35, 50, False), (34, 50, True)])
def test_a_script_under_70_percent_of_its_target_is_flagged(words, target, short):
    brief = SimpleNamespace(target_words=target)
    assert video_tab.is_thin(SimpleNamespace(word_count=words), brief) is short
    assert video_tab.is_thin(SimpleNamespace(word_count=words), SimpleNamespace(target_words=0)) is False


def test_the_review_page_warns_above_the_script_when_it_is_short(folder):
    record = _planned(folder)
    words = record.script.word_count
    for target, flagged in ((words * 3, True), (words, False)):
        record.brief.target_words = target
        repository.save_generation(record)
        warnings = [w.value for w in _app(video=record.video_id).run().warning if "This script is short" in w.value]
        assert bool(warnings) is flagged
        if flagged:
            assert f"{words} words, {words * 100 // target}% of the {target}" in warnings[0]
