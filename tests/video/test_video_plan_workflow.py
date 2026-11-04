"""The plan graph end to end: nodes 1-5 on a synthetic property, offline, saved."""
from __future__ import annotations

import re

import pytest

from src.core.config import settings
from src.core.video_state import VideoState
from src.data import video_repository as repository
from src.data.video_schemas import Storyboard, VideoScript
from src.workflows import video as workflow
from tests.video.helpers import make_video_property


@pytest.fixture(autouse=True)
def _database(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "videos.db"), raising=False)


@pytest.fixture
def refuse_models(monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a model was asked")

    for target in ("src.agents.brochure.property_intelligence.call_with_fallback",
                   "src.agents.brochure.vision.call_with_fallback"):
        monkeypatch.setattr(target, refuse)
    return calls


@pytest.fixture
def folder(tmp_path):
    return make_video_property(tmp_path / "properties")


def test_the_plan_is_made_saved_and_written_beside_the_record(folder, refuse_models):
    record = workflow.run_plan_workflow(folder, offline=True)

    assert record.status == "planned" and record.error is None
    assert re.fullmatch(r"vid_[0-9a-f]{10}", record.video_id)
    assert record.property_id == "AURELIA-HEIGHTS"
    out = repository.video_output_dir(record.video_id)
    assert record.output_dir == str(out)
    assert VideoScript.model_validate_json((out / "script.json").read_text(encoding="utf-8")) == record.script
    assert Storyboard.model_validate_json((out / "storyboard.json").read_text(encoding="utf-8")) == record.storyboard
    assert record.video_path is None and record.cost_usd == 0
    assert refuse_models == []


def test_the_saved_record_reads_back_whole(folder, refuse_models):
    record = workflow.run_plan_workflow(folder, offline=True)
    back = repository.get_generation(record.video_id)
    assert back.status == "planned" and back.script == record.script and back.storyboard == record.storyboard
    assert back.brief.name == "Aurelia Heights" and back.duration_s == 15 and back.aspect == "9:16"
    assert [r.video_id for r in repository.list_generations(record.property_id)] == [record.video_id]


def test_every_step_is_recorded_in_order_and_succeeds(folder, refuse_models):
    record = workflow.run_plan_workflow(folder, offline=True)
    steps = record.workflow_evaluation.steps
    assert [s.node for s in steps] == workflow._PLAN_STEPS == ["brief", "fact_ledger", "asset_audit", "script",
                                                                "storyboard"]
    assert all(s.status == "success" for s in steps)


def test_every_agent_leaves_a_report(folder, refuse_models):
    record = workflow.run_plan_workflow(folder, offline=True)
    assert set(record.agent_reports) == {"brief", "fact_ledger", "asset_audit", "script", "storyboard"}
    assert "15 s, 9:16, 1080x1920 (final)" in record.agent_reports["brief"][0]
    assert record.notes and record.generated_shots == []


def test_graph_node_ids_carry_the_prefix_and_state_keys_survive(folder, refuse_models):
    nodes = set(workflow.get_plan_graph().get_graph().nodes)
    assert {"n_brief", "n_fact_ledger", "n_asset_audit", "n_script", "n_storyboard"} <= nodes
    assert not {"brief", "script", "storyboard"} & nodes

    state = workflow.get_plan_graph().invoke({"video_id": "vid_x", "property_dir": str(folder), "offline": True,
                                              "video_overrides": {}, "agent_reports": {}})
    declared = set(VideoState.__annotations__)
    returned = {"property_input", "images", "documents_text", "video_input", "brief", "brand_kit",
                "property_knowledge", "image_analyses", "ranked_assets", "image_plan", "asset_notes", "script",
                "concept", "storyboard", "agent_reports"}
    assert returned <= declared and returned <= set(state)         # LangGraph drops what is not declared


def test_command_line_options_reach_the_brief(folder, refuse_models):
    record = workflow.run_plan_workflow(folder, {"duration_s": 10, "aspect": "1:1", "model": "kenburns"}, offline=True)
    assert (record.duration_s, record.aspect, record.provider) == (10, "1:1", "kenburns")
    assert record.storyboard.total_seconds == pytest.approx(10.0, abs=0.05)


def test_a_draft_plans_for_the_small_size(folder, refuse_models):
    record = workflow.run_plan_workflow(folder, offline=True, draft=True)
    assert (record.brief.width, record.brief.height, record.brief.quality) == (540, 960, "draft")


def test_offline_no_model_is_asked_even_when_a_key_is_set(folder, refuse_models, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "a-key-that-must-not-be-used", raising=False)
    record = workflow.run_plan_workflow(folder, offline=True)
    assert record.status == "planned" and refuse_models == []


def test_with_no_key_at_all_no_model_is_asked_either(folder, refuse_models):
    assert settings.gemini_api_key == "" and settings.openrouter_api_key == ""
    assert workflow.run_plan_workflow(folder, offline=False).status == "planned"
    assert refuse_models == []


def test_the_storyboard_goes_to_the_approval_step_and_what_comes_back_is_used(folder, refuse_models):
    seen = []

    def approve(board: Storyboard) -> Storyboard:
        seen.append(board)
        edited = board.model_copy(deep=True)
        edited.shots[1].camera = "tilt_up"
        return edited

    record = workflow.run_plan_workflow(folder, offline=True, approve=approve)

    assert len(seen) == 1 and seen[0].shots[1].camera != "tilt_up"
    assert record.status == "planned" and record.storyboard.shots[1].camera == "tilt_up"
    out = repository.video_output_dir(record.video_id)
    assert '"tilt_up"' in (out / "storyboard.json").read_text(encoding="utf-8")


def test_by_default_the_plan_is_approved_as_it_stands(folder, refuse_models):
    board = Storyboard()
    assert workflow.auto_approve(board) is board


def test_an_approved_storyboard_that_breaks_the_rules_fails_the_run(folder, refuse_models):
    def approve(board: Storyboard) -> Storyboard:
        edited = board.model_copy(deep=True)
        edited.shots[1].image_id = "IMG999"
        return edited

    record = workflow.run_plan_workflow(folder, offline=True, approve=approve)

    assert record.status == "failed" and "The approved storyboard is not valid" in record.error
    assert "does not exist (IMG999)" in record.error
    assert repository.get_generation(record.video_id).status == "failed"


def test_a_property_with_no_pictures_fails_in_words(tmp_path, refuse_models):
    folder = make_video_property(tmp_path / "properties")
    for picture in (folder / "images").iterdir():
        picture.unlink()

    record = workflow.run_plan_workflow(folder, offline=True)

    assert record.status == "failed" and "No images found" in record.error
    assert [(s.node, s.status) for s in record.workflow_evaluation.steps] == [("brief", "failed")]
    assert record.script is None and record.output_dir is None


def test_a_folder_that_is_not_a_property_fails_in_words(tmp_path, refuse_models):
    record = workflow.run_plan_workflow(tmp_path / "properties" / "NOT-A-PROPERTY", offline=True)
    assert record.status == "failed" and "Missing property.json" in record.error
    assert record.property_id == "NOT-A-PROPERTY"


def test_a_bad_option_fails_the_run_with_the_valid_choices(folder, refuse_models):
    record = workflow.run_plan_workflow(folder, {"duration_s": 20}, offline=True)
    assert record.status == "failed" and "[10, 15, 30, 45, 60]" in record.error


def test_each_run_gets_its_own_id_and_folder(folder, refuse_models):
    first = workflow.run_plan_workflow(folder, offline=True)
    second = workflow.run_plan_workflow(folder, offline=True)
    assert first.video_id != second.video_id and first.output_dir != second.output_dir
    assert first.script == second.script                           # but the same plan
