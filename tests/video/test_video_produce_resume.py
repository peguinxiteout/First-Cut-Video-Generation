"""Plan in one process, produce in another: plan_state.json, run_produce_workflow(video_id), and the
progress saved after every node."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.agents.video import assets as asset_audit
from src.agents.video import brief as brief_agent
from src.agents.video import fact_ledger
from src.core.config import PROJECT_ROOT
from src.core.exceptions import WorkflowError
from src.data import video_plan_state as plan_state
from src.data import video_repository as repository
from src.data.video_schemas import PLAN_STATE_FILE, PLAN_STATE_VERSION, VideoGenerationRecord, VideoRequest
from src.workflows import video as workflow
from tests.video.helpers import make_video_property


@pytest.fixture(autouse=True)
def _database(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "videos.db"), raising=False)


@pytest.fixture(autouse=True)
def refuse_models(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a model was asked")

    for target in ("src.agents.brochure.property_intelligence.call_with_fallback",
                   "src.agents.brochure.vision.call_with_fallback"):
        monkeypatch.setattr(target, refuse)


@pytest.fixture
def folder(tmp_path):
    return make_video_property(tmp_path / "properties")


@pytest.fixture
def watched(monkeypatch):
    """Every progress write, with what the database said right after it."""
    seen: list[tuple[str, str, int]] = []
    real = repository.update_progress

    def watch(video_id, **kwargs):
        done = real(video_id, **kwargs)
        row = repository.get_summary(video_id)
        seen.append((row.status, row.current_node, row.steps_done))
        return done

    monkeypatch.setattr(repository, "update_progress", watch)
    return seen


# -- plan_state.json ----------------------------------------------------------------------------------

def test_the_plan_writes_plan_state_beside_the_script(folder):
    record = workflow.run_plan_workflow(folder, offline=True, draft=True)
    out = Path(record.output_dir)
    path = out / PLAN_STATE_FILE
    assert path.is_file() and (out / "script.json").is_file()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == PLAN_STATE_VERSION == 1
    assert set(data) == {"version", "video_id", "property_dir", "offline", "draft", "refresh_ledger", "overrides",
                         "video_input", "property_input", "property_knowledge", "brand_kit", "image_analyses",
                         "ranked_assets", "image_plan", "asset_notes", "concept"}
    assert data["video_id"] == record.video_id and data["draft"] is True and data["offline"] is True
    text = path.read_text(encoding="utf-8").lower()
    assert "api_key" not in text and "token" not in text


def test_the_saved_state_reads_back_whole(folder):
    record = workflow.run_plan_workflow(folder, offline=True)
    saved = plan_state.load(record.video_id)

    assert saved.property_input.name == "Aurelia Heights" and saved.brand_kit.name == "Aurelia Heights"
    assert saved.property_knowledge.facts and saved.ranked_assets
    assert Path(saved.property_dir) == folder.resolve()
    assert all(Path(a.path).is_file() for a in saved.ranked_assets)
    assert saved.brand_kit.logo_path and Path(saved.brand_kit.logo_path).is_file()
    plan = plan_state.plan_from_data(saved.image_plan)
    assert all(isinstance(k, int) for k in plan.members)
    assert set(plan.cluster_of) <= {a.image_id for a in saved.ranked_assets}


def test_paths_under_the_project_are_stored_relative_and_come_back_absolute():
    inside = PROJECT_ROOT / "data" / "properties" / "X" / "images" / "a.jpg"
    assert plan_state._relative(str(inside)) == "data/properties/X/images/a.jpg"
    assert Path(plan_state._absolute("data/properties/X/images/a.jpg")) == inside.resolve()
    elsewhere = str(Path(PROJECT_ROOT.anchor) / "elsewhere" / "a.jpg")
    assert plan_state._relative(elsewhere) == elsewhere and plan_state._relative(None) is None


def test_a_plan_job_record_made_beforehand_is_the_one_planned(folder, watched):
    made = VideoGenerationRecord(video_id="vid_made000001", property_id=folder.name, status="planning",
                                 request=VideoRequest(property_id=folder.name, offline=True, draft=True))
    repository.init_db()
    repository.begin_job(made)

    record = workflow.run_plan_workflow(folder, offline=True, draft=True, video_id="vid_made000001")

    assert record.video_id == "vid_made000001" and record.status == "planned"
    assert [r.video_id for r in repository.list_summaries()] == ["vid_made000001"]
    back = repository.get_generation("vid_made000001")
    assert back.request.offline is True and back.request.draft is True and back.pid is None
    assert (Path(back.output_dir) / PLAN_STATE_FILE).is_file()


def test_planning_an_unknown_video_id_is_an_error(folder):
    with pytest.raises(WorkflowError, match="no video"):
        workflow.run_plan_workflow(folder, offline=True, video_id="vid_nothere001")


# -- progress -------------------------------------------------------------------------------------------

def test_the_status_and_current_node_are_saved_before_and_after_every_node(folder, watched):
    record = workflow.run_plan_workflow(folder, offline=True)

    nodes = workflow._PLAN_STEPS
    assert [(n, d) for _, n, d in watched] == [pair for i, n in enumerate(nodes) for pair in ((n, i), (n, i + 1))]
    assert {status for status, _, _ in watched} == {"planning"}
    assert record.status == "planned" and record.steps_done == 5 and record.current_node == ""
    final = repository.get_summary(record.video_id)
    assert (final.status, final.steps_done, final.current_node, final.pid) == ("planned", 5, "", None)


def test_a_failed_node_leaves_the_steps_so_far_and_a_failed_record(folder, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the script step broke")

    monkeypatch.setattr(workflow.script_agent, "run", broken)
    record = workflow.run_plan_workflow(folder, offline=True)

    assert record.status == "failed" and "the script step broke" in record.error
    assert [(s.node, s.status) for s in record.workflow_evaluation.steps][-1] == ("script", "failed")
    assert repository.get_generation(record.video_id).steps_done == 3
    assert not (Path(repository.video_output_dir(record.video_id)) / PLAN_STATE_FILE).exists()


# -- run_produce_workflow: what it refuses --------------------------------------------------------------

def _planned(folder) -> VideoGenerationRecord:
    return workflow.run_plan_workflow(folder, offline=True, draft=True)


@contextmanager
def _no_replanning(monkeypatch):
    """Inside the block the planning agents refuse to run: the produce half must use what was saved."""
    def refuse(*args, **kwargs):
        raise AssertionError("the plan was made again")

    with monkeypatch.context() as patch:
        for module in (brief_agent, fact_ledger, asset_audit):
            patch.setattr(module, "run", refuse)
        yield


def test_a_missing_plan_state_fails_the_job_and_does_not_plan_again(folder, monkeypatch):
    planned = _planned(folder)
    (Path(planned.output_dir) / PLAN_STATE_FILE).unlink()
    with _no_replanning(monkeypatch):
        record = workflow.run_produce_workflow(planned.video_id)

    assert record.status == "failed" and PLAN_STATE_FILE in record.error and "missing" in record.error
    assert record.video_path is None and [s.node for s in record.workflow_evaluation.steps] == workflow._PLAN_STEPS
    assert record.script == planned.script and record.storyboard == planned.storyboard
    assert repository.get_generation(planned.video_id).status == "failed"


def test_a_plan_state_from_another_version_fails_the_job(folder, monkeypatch):
    planned = _planned(folder)
    path = Path(planned.output_dir) / PLAN_STATE_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = PLAN_STATE_VERSION + 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with _no_replanning(monkeypatch):
        record = workflow.run_produce_workflow(planned.video_id)

    assert record.status == "failed" and f"version {PLAN_STATE_VERSION + 1}" in record.error
    assert "has to be made again" in record.error


def test_an_unreadable_plan_state_fails_the_job(folder):
    planned = _planned(folder)
    (Path(planned.output_dir) / PLAN_STATE_FILE).write_text("{ not json", encoding="utf-8")

    record = workflow.run_produce_workflow(planned.video_id)

    assert record.status == "failed" and "cannot be read" in record.error


def test_only_a_planned_video_can_be_produced(folder):
    with pytest.raises(WorkflowError, match="no video"):
        workflow.run_produce_workflow("vid_nothere001")
    planned = _planned(folder)
    repository.save_generation(planned.model_copy(update={"status": "passed"}))
    with pytest.raises(WorkflowError, match="only a planned video"):
        workflow.run_produce_workflow(planned.video_id)


# -- the real thing -------------------------------------------------------------------------------------

@pytest.mark.slow
def test_producing_from_the_saved_plan_makes_the_same_video_as_a_one_shot_run(folder, watched, monkeypatch):
    planned = _planned(folder)
    state_before = (Path(planned.output_dir) / PLAN_STATE_FILE).read_bytes()
    watched.clear()
    with _no_replanning(monkeypatch):
        done = workflow.run_produce_workflow(planned.video_id)

    assert done.video_id == planned.video_id and done.output_dir == planned.output_dir
    assert done.status in ("passed", "warned") and done.error is None
    out = Path(done.output_dir)
    for name in ("video.mp4", "thumbnail.jpg", "captions.srt", "qa.json"):
        assert (out / name).is_file(), name
    assert [(s.node, s.status) for s in done.workflow_evaluation.steps] == [(n, "success") for n in workflow._STEP_NAMES]
    assert (out / PLAN_STATE_FILE).read_bytes() == state_before
    assert done.agent_reports.keys() >= {"brief", "script", "storyboard", "voice_music", "gate"}
    assert done.brief == planned.brief and done.script == planned.script

    assert {status for status, _, _ in watched} == {"producing"}
    assert [n for _, n, d in watched][:2] == ["voice_music", "voice_music"]
    assert watched[-1] == ("producing", "gate", 10) and watched[0][2] == 5

    again = repository.get_generation(done.video_id)
    assert again.status == done.status and again.pid is None and again.steps_done == 10

    one_shot = workflow.run_video_workflow(folder, offline=True, draft=True)
    assert one_shot.video_id != done.video_id
    assert one_shot.script == done.script and one_shot.storyboard == done.storyboard
    assert one_shot.status == done.status
