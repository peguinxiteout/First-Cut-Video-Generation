"""The Video tab: the form, the model selector, starting a plan, progress, approval and the result.
Jobs are stubbed; the database and the data folder are in a temp folder (the app calls init_db on the
database in Settings, so database_path is moved before every run)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.core.config import settings  # noqa: E402
from src.core.exceptions import IngestionError, JobBusyError, JobError  # noqa: E402
from src.data import video_repository as repository  # noqa: E402
from src.data.schemas import AgentStepResult  # noqa: E402
from src.data.video_schemas import (  # noqa: E402
    PLAN_STATE_FILE, VideoEvaluationResult, VideoGate, VideoGenerationRecord, VideoInput, VideoRequest, VideoSummary,
)
from src.ui import video_tab  # noqa: E402
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


# -- the page ----------------------------------------------------------------------------------------------

def test_the_video_tab_loads_with_the_property_and_the_brief(folder):
    at = _app().run()

    assert not at.exception
    assert at.radio(key="solution").options == ["Brochure", "Blog post", "Video"]
    assert at.radio(key="solution").value == "Video"
    assert not any("Project name" in w.label for w in at.text_input)           # not the brochure form
    assert at.selectbox(key="vt_property").options == ["Aurelia Heights (AURELIA-HEIGHTS)"]
    assert at.selectbox(key="vt_property").value == PROPERTY
    labels = [w.label for w in (*at.selectbox, *at.text_input, *at.text_area, *at.checkbox)]
    for expected in ("Length", "Shape", "What it is for", "Who is it for?", "Key highlights to feature (one per line)",
                     "Call to action", "Creative direction", "Tone", "Music mood", "Voice", "Opening picture",
                     "Print the narration on screen", "Light colour grade", "Allow generated frames",
                     "Pronunciations (one per line: word=how it is said)", "Offline (no models, no cost)",
                     "Draft quality (fast)"):
        assert expected in labels, expected
    assert at.checkbox(key=_key("allow_generated")).value is False
    assert at.checkbox(key=_key("offline")).value is False and at.checkbox(key=_key("draft")).value is False
    assert at.selectbox(key=_key("duration_s")).options == [f"{s} seconds" for s in (10, 15, 30, 45, 60)]
    assert at.selectbox(key=_key("duration_s")).value == 15
    assert at.selectbox(key=_key("objective")).options == ["Awareness", "Site visit", "Launch", "Enquiry"]


def test_only_folders_with_a_property_json_are_offered_and_they_are_named(folder, tmp_path):
    (folder.parent / "NOT-A-PROPERTY").mkdir()
    (folder.parent / "stray.txt").write_text("x", encoding="utf-8")
    assert video_tab.property_choices() == [(PROPERTY, "Aurelia Heights (AURELIA-HEIGHTS)")]
    assert _app().run().selectbox(key="vt_property").options == ["Aurelia Heights (AURELIA-HEIGHTS)"]


def test_with_no_property_the_page_says_what_to_do(tmp_path):
    at = _app().run()
    assert not at.exception and "no property to make a video of" in _shown(at)


def test_the_form_starts_from_the_property_file(folder):
    (folder / "video_input.json").write_text(json.dumps({
        "duration_s": 10, "aspect": "1:1", "audience": "young families", "objective": "launch",
        "highlights": ["Swimming Pool", "Gym"], "tone": "warm", "burn_captions": False,
        "pronunciations": {"Anamika": "Ah-nah-mih-kah"}}), encoding="utf-8")
    at = _app().run()

    assert at.selectbox(key=_key("duration_s")).value == 10 and at.selectbox(key=_key("aspect")).value == "1:1"
    assert at.selectbox(key=_key("objective")).value == "launch" and at.selectbox(key=_key("tone")).value == "warm"
    assert at.text_input(key=_key("audience")).value == "young families"
    assert at.text_area(key=_key("highlights")).value == "Swimming Pool\nGym"
    assert at.text_area(key=_key("pronunciations")).value == "Anamika=Ah-nah-mih-kah"
    assert at.checkbox(key=_key("burn_captions")).value is False and at.checkbox(key=_key("grade")).value is True


def _rows(at) -> dict[str, str]:
    """The "not available yet" rows of the model list, by the label each starts with."""
    rows = [str(c.value) for c in at.caption if " · not available yet" in str(c.value)]
    return {row.split(" · ")[0]: row for row in rows}


def test_the_model_selector_offers_what_can_run_and_lists_the_rest_as_not_available_yet(folder):
    at = _app().run()

    radio = at.radio(key=_key("model"))
    assert radio.options[0].startswith("Auto") and radio.value == "auto"
    assert len(radio.options) == 2 and radio.options[1].startswith("Pan-and-zoom") and "free" in radio.options[1]
    assert not [c for c in at.checkbox if c.key and c.key.startswith("vt_na_")]          # text rows, not boxes
    assert "**Not available yet**" in [str(m.value) for m in at.markdown]
    rows = _rows(at)
    assert list(rows) == ["MiniMax H3 Max", "Kling 2.5 Turbo Pro", "Seedance 2.0", "Magic Hour Wan 2.2 I2V",
                          "8Scale Wan 2.2 14B I2V", "Nodaro Seedance 2 Fast", "Nodaro Veo 3.1 Lite",
                          "Eternal AI Wan 2.2 I2V"]
    assert "via fal.ai" in rows["MiniMax H3 Max"] and "about $0.45" in rows["MiniMax H3 Max"]    # 0.03 a second, 15 s
    assert "price not set" in rows["Seedance 2.0"]
    assert at.radio(key=_key("model")).disabled is False
    at.checkbox(key=_key("offline")).check().run()
    assert at.radio(key=_key("model")).disabled is True


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
    at.selectbox(key=_key("duration_s")).select(60).run()
    rows = _rows(at)
    assert "about $1.80" in rows["MiniMax H3 Max"]
    assert "from about $0.20" in rows["8Scale Wan 2.2 14B I2V"] and "from about $0.30" in rows["Eternal AI Wan 2.2 I2V"]
    assert "have not been rendered" in _shown(at)


def test_start_sends_every_option_with_the_run_flags(folder, calls):
    at = _app().run()
    at.text_input(key=_key("audience")).set_value("young families")
    at.text_area(key=_key("highlights")).set_value("Gym\nSwimming Pool")
    at.selectbox(key=_key("tone")).select("warm")
    at.checkbox(key=_key("draft")).check()
    at.button(key="vt_start").click().run()

    assert not at.exception
    (property_id, overrides, flags), = calls["plan"]
    assert property_id == PROPERTY and flags == {"offline": False, "draft": True}
    assert overrides["audience"] == "young families" and overrides["highlights"] == ["Gym", "Swimming Pool"]
    assert overrides["tone"] == "warm" and overrides["model"] == "auto" and set(overrides) == set(VideoInput.model_fields)
    assert at.query_params["video"] == ["vid_stub000001"] or at.query_params["video"] == "vid_stub000001"


def test_offline_is_passed_on(folder, calls):
    at = _app().run()
    at.checkbox(key=_key("offline")).check()
    at.button(key="vt_start").click().run()
    assert calls["plan"][0][2] == {"offline": True, "draft": False}


def test_a_busy_job_is_named_with_its_step_and_can_be_opened(folder, monkeypatch):
    _seed("vid_busy0000001", "planning", current_node="script", steps_done=3)
    repository.set_pid("vid_busy0000001", os.getpid())

    def busy(*args, **kwargs):
        raise JobBusyError("Video vid_busy0000001 is still planning.")

    monkeypatch.setattr(video_jobs, "start_plan", busy)
    at = _app().run()
    at.button(key="vt_start").click().run()

    warning = " ".join(w.value for w in at.warning)
    assert "vid_busy0000001" in warning and "write the script" in warning and "planning" in warning
    assert "Only one video is made at a time" in warning
    at.button(key="vt_open_vid_busy0000001").click().run()
    assert at.query_params["video"] in ("vid_busy0000001", ["vid_busy0000001"])
    assert "Planning the video" in _shown(at)


def test_a_refused_start_is_explained(folder, monkeypatch):
    def refuse(*args, **kwargs):
        raise JobError("The video options are not valid: duration_s: x.")

    monkeypatch.setattr(video_jobs, "start_plan", refuse)
    at = _app().run()
    at.button(key="vt_start").click().run()
    assert "The video options are not valid" in " ".join(e.value for e in at.error)


def test_save_as_defaults_writes_the_file(folder):
    at = _app().run()
    at.text_input(key=_key("audience")).set_value("retirees")
    at.button(key="vt_save").click().run()

    assert json.loads((folder / "video_input.json").read_text(encoding="utf-8"))["audience"] == "retirees"
    assert "Saved to video_input.json" in _shown(at)


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
    if record.notes:
        assert any("Notes about this plan" in e.label for e in at.expander)


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
    assert overrides == {"audience": "families"} and flags == {"offline": True, "draft": True, "refresh_ledger": False}
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
