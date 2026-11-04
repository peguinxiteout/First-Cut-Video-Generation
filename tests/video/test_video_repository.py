from __future__ import annotations

import sqlite3

import pytest

from src.core.types import QualityGateResult
from src.data import video_repository as repo
from src.data.schemas import AgentStepResult, WorkflowEvaluation
from src.data.video_schemas import (
    Shot, Storyboard, VideoBrief, VideoEvaluationResult, VideoGenerationRecord, VideoMeasurements,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "v.db"), raising=False)
    repo.init_db()
    return tmp_path / "v.db"


def _record(video_id="vid_0000000001", property_id="P1", **extra) -> VideoGenerationRecord:
    fields = dict(
        video_id=video_id, property_id=property_id, status="completed", video_path="/x/video.mp4",
        thumbnail_path="/x/thumbnail.jpg", duration_s=15, aspect="9:16", provider="auto", cost_usd=0.21,
        brief=VideoBrief(property_id=property_id, name="Test Residences"),
        storyboard=Storyboard(shots=[Shot(index=0, seconds=3.0)]),
        measurements=VideoMeasurements(duration_s=15.04, width=1080, height=1920, vcodec="h264", acodec="aac"),
        evaluation=VideoEvaluationResult(quality_score=88.0, gate_result=QualityGateResult.PASS,
                                         checks={"duration": "pass"}),
        workflow_evaluation=WorkflowEvaluation(steps=[AgentStepResult(node="brief", latency_ms=4)]))
    fields.update(extra)
    return VideoGenerationRecord(**fields)


def test_full_record_round_trips_and_human_review_stays_authoritative(db):
    repo.save_generation(_record())
    repo.update_human_review("vid_0000000001", status="approved", comment="Ready to post")

    back = repo.get_generation("vid_0000000001")
    assert back.video_path == "/x/video.mp4" and back.cost_usd == 0.21
    assert back.brief.name == "Test Residences"
    assert back.storyboard.shots[0].seconds == 3.0
    assert back.evaluation.gate_result == QualityGateResult.PASS and back.evaluation.checks == {"duration": "pass"}
    assert back.measurements.vcodec == "h264"
    assert back.workflow_evaluation.steps[0].node == "brief"
    assert back.human_review.status == "approved" and back.human_review.comment == "Ready to post"


def test_saving_again_updates_the_row_and_keeps_the_review(db):
    repo.save_generation(_record(status="running"))
    repo.update_human_review("vid_0000000001", status="revision_requested", comment="Shorter")
    repo.save_generation(_record())

    back = repo.get_generation("vid_0000000001")
    assert back.status == "completed"
    assert back.human_review.status == "revision_requested"


def test_an_unknown_id_is_none(db):
    assert repo.get_generation("vid_nope") is None


def test_list_filters_by_property(db):
    repo.save_generation(_record("vid_a", "P1"))
    repo.save_generation(_record("vid_b", "P2"))
    repo.save_generation(_record("vid_c", "P1"))

    assert {r.video_id for r in repo.list_generations("P1")} == {"vid_a", "vid_c"}
    assert len(repo.list_generations()) == 3
    assert [r.video_id for r in repo.list_generations("P2")] == ["vid_b"]


def test_a_row_whose_record_json_no_longer_validates_reads_from_the_columns(db, caplog):
    repo.save_generation(_record())
    conn = sqlite3.connect(db)
    conn.execute("UPDATE video_generations SET record_json = ?",
                 ('{"video_id": "vid_0000000001", "duration_s": "soon"}',))
    conn.commit()
    conn.close()

    with caplog.at_level("WARNING", logger="src.data.video_repository"):
        back = repo.get_generation("vid_0000000001")
    assert back.video_path == "/x/video.mp4" and back.duration_s == 15 and back.cost_usd == 0.21
    assert back.workflow_evaluation.steps[0].node == "brief"
    assert "vid_0000000001" in caplog.text and "duration_s" in caplog.text


def test_init_db_can_run_twice(db):
    repo.init_db()
    repo.save_generation(_record())
    repo.init_db()
    assert repo.get_generation("vid_0000000001") is not None


def test_the_output_folder_is_under_the_current_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.data_dir", str(tmp_path / "elsewhere"), raising=False)
    folder = repo.video_output_dir("vid_0123456789")
    assert folder.name == "vid_0123456789"
    assert folder.parent.name == "generated_videos"
    assert str(folder).startswith(str((tmp_path / "elsewhere").resolve()))


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", "", ".hidden"])
def test_an_id_that_is_not_a_single_folder_name_is_refused(bad):
    with pytest.raises(ValueError):
        repo.video_output_dir(bad)
