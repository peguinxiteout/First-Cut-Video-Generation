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


# -- jobs: progress, the one-at-a-time guard, history, the sweep ---------------------------------------

import threading  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

from src.core.exceptions import JobBusyError, JobError  # noqa: E402


def _job(video_id="vid_job0000001", status="planning", **extra) -> VideoGenerationRecord:
    return VideoGenerationRecord(video_id=video_id, property_id="P1", status=status, **extra)


def test_a_new_database_gets_the_job_columns_and_an_old_one_is_upgraded(tmp_path, monkeypatch):
    old = tmp_path / "old.db"
    with sqlite3.connect(old) as conn:
        conn.executescript(repo._SCHEMA)
        conn.execute("INSERT INTO video_generations (video_id, property_id, created_at, status) "
                     "VALUES ('vid_old', 'P1', '2026-11-01T10:00:00+00:00', 'completed')")
    monkeypatch.setattr("src.core.config.settings.database_path", str(old), raising=False)
    assert repo.get_generation("vid_old").current_node == ""          # readable before the upgrade
    repo.init_db()
    with sqlite3.connect(old) as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(video_generations)")}
    assert set(repo._NEW_GENERATION_COLUMNS) <= columns
    assert repo.get_summary("vid_old").status == "completed"


def test_a_second_job_is_refused_while_one_is_planning_or_producing(db):
    repo.begin_job(_job("vid_a"))
    with pytest.raises(JobBusyError, match="vid_a"):
        repo.begin_job(_job("vid_b"))
    assert repo.get_generation("vid_b") is None
    repo.save_generation(_job("vid_a", status="planned"))
    repo.begin_job(_job("vid_b"))                                       # a planned video holds no slot
    repo.save_generation(_job("vid_b", status="producing"))
    with pytest.raises(JobBusyError):
        repo.begin_job(_job("vid_c"))


def test_the_check_and_the_insert_are_one_transaction(db):
    """Callers racing for the slot: one wins, whatever the timing."""
    results, barrier = [], threading.Barrier(6)

    def start(n):
        barrier.wait()
        try:
            repo.begin_job(_job(f"vid_race{n}"))
            results.append("started")
        except JobBusyError:
            results.append("busy")

    threads = [threading.Thread(target=start, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["busy"] * 5 + ["started"]
    assert len(repo.list_summaries()) == 1


def test_the_write_lock_is_taken_before_the_check(db):
    """BEGIN IMMEDIATE: while one process is between its check and its insert, another cannot start its own."""
    with repo._exclusive():
        other = sqlite3.connect(db, timeout=0.2, isolation_level=None)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("BEGIN IMMEDIATE")
        finally:
            other.close()


def test_produce_starts_only_from_planned_and_claims_the_slot(db):
    with pytest.raises(JobError, match="no video"):
        repo.begin_produce("vid_nope")
    repo.save_generation(_job("vid_a", status="failed"))
    with pytest.raises(JobError, match="only a planned video"):
        repo.begin_produce("vid_a")
    repo.save_generation(_job("vid_a", status="planned"))
    repo.save_generation(_job("vid_b", status="planning"))
    with pytest.raises(JobBusyError, match="vid_b"):
        repo.begin_produce("vid_a")
    repo.save_generation(_job("vid_b", status="passed"))
    assert repo.begin_produce("vid_a").status == "producing"
    with pytest.raises(JobError, match="only a planned video"):
        repo.begin_produce("vid_a")                                     # not twice


def test_progress_is_a_cheap_write_that_leaves_record_json_alone(db):
    repo.save_generation(_job())
    with repo.get_connection() as conn:
        before = conn.execute("SELECT record_json FROM video_generations").fetchone()[0]
    steps = [AgentStepResult(node="brief", latency_ms=4),
             AgentStepResult(node="fact_ledger", status="failed", latency_ms=2, error="x")]
    assert repo.update_progress("vid_job0000001", current_node="asset_audit", steps=steps)
    with repo.get_connection() as conn:
        after = conn.execute("SELECT record_json FROM video_generations").fetchone()[0]
    assert after == before

    record = repo.get_generation("vid_job0000001")
    assert record.current_node == "asset_audit" and record.steps_done == 1       # only a success counts
    assert [s.node for s in record.workflow_evaluation.steps] == ["brief", "fact_ledger"]
    assert record.updated_at is not None
    assert repo.get_summary("vid_job0000001").steps_done == 1


def test_progress_cannot_revive_or_undo_a_finished_row(db):
    repo.save_generation(_job(status="passed"))
    assert repo.update_progress("vid_job0000001", current_node="gate") is False
    repo.save_generation(_job("vid_c"))
    assert repo.mark_cancelled("vid_c") is True
    assert repo.update_progress("vid_c", status="producing") is False
    repo.save_generation(_job("vid_c", status="failed"))                # a late save by the killed job
    assert repo.get_generation("vid_c").status == "cancelled"
    assert repo.mark_cancelled("vid_c") is False


def test_the_pid_is_kept_while_running_and_cleared_when_done(db):
    repo.save_generation(_job())
    repo.set_pid("vid_job0000001", 4242)
    repo.save_generation(_job())                                        # a save from the job, no pid on it
    assert repo.get_generation("vid_job0000001").pid == 4242
    repo.save_generation(_job(status="passed"))
    assert repo.get_generation("vid_job0000001").pid is None


def test_the_history_is_listed_from_the_columns_without_reading_record_json(db):
    for n, status in enumerate(["passed", "failed", "planned", "passed"]):
        repo.save_generation(_record(f"vid_{n}", "P1" if n % 2 == 0 else "P2", status=status,
                                     created_at=datetime(2026, 11, 1, 10, n, tzinfo=timezone.utc)))
    with repo.get_connection() as conn:
        conn.execute("UPDATE video_generations SET record_json = 'not json at all'")

    rows = repo.list_summaries()
    assert [r.video_id for r in rows] == ["vid_3", "vid_2", "vid_1", "vid_0"]          # newest first
    assert [r.video_id for r in repo.list_summaries(limit=2, offset=1)] == ["vid_2", "vid_1"]
    assert {r.video_id for r in repo.list_by_status("passed")} == {"vid_0", "vid_3"}
    assert {r.video_id for r in repo.list_by_status("planned", "failed")} == {"vid_1", "vid_2"}
    assert [r.video_id for r in repo.list_summaries(property_id="P2")] == ["vid_3", "vid_1"]
    first = rows[-1]
    assert (first.gate_result, first.quality_score, first.duration_s, first.cost_usd) == ("pass", 88.0, 15, 0.21)
    assert first.video_path == "/x/video.mp4" and first.human_review_status == "pending"


def test_the_sweep_fails_a_job_whose_process_is_gone_and_leaves_the_rest(db):
    repo.save_generation(_job("vid_dead"))
    repo.set_pid("vid_dead", 111)
    repo.save_generation(_job("vid_alive", status="producing"))
    repo.set_pid("vid_alive", 222)
    repo.save_generation(_job("vid_fresh"))                              # no pid yet, just made
    repo.save_generation(_job("vid_done", status="passed"))

    assert repo.sweep_interrupted(lambda pid: pid == 222) == ["vid_dead"]

    dead = repo.get_generation("vid_dead")
    assert dead.status == "failed" and "interrupted" in dead.error and "interrupted" in dead.notes
    assert dead.pid is None
    assert [repo.get_generation(v).status for v in ("vid_alive", "vid_fresh", "vid_done")] == [
        "producing", "planning", "passed"]


def test_a_row_with_no_pid_is_only_swept_when_it_has_gone_quiet(db):
    repo.save_generation(_job("vid_old"))
    with repo.get_connection() as conn:
        conn.execute("UPDATE video_generations SET updated_at = ?",
                     ((datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),))
    assert repo.sweep_interrupted(lambda pid: True) == ["vid_old"]


def test_a_job_knows_when_it_started_and_a_produce_job_starts_again(db):
    assert repo.get_summary("vid_none") is None
    repo.begin_job(_job("vid_a"))
    first = repo.get_summary("vid_a").started_at
    assert first is not None
    repo.save_generation(_job("vid_a", status="planned"))
    assert repo.get_summary("vid_a").started_at == first                # a save does not move it
    time.sleep(0.01)
    repo.begin_produce("vid_a")
    assert repo.get_summary("vid_a").started_at > first
