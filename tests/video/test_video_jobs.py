"""Background video jobs: one at a time, the record made before the process starts, status read back from the
database, the sweep, cancel. The launch is stubbed except in the tests that start a real process."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import generate_video, run_video_job
from src.core.config import settings
from src.core.exceptions import JobBusyError, JobError
from src.data import video_repository as repository
from src.data.video_schemas import PLAN_STATE_FILE
from src.workflows import video_jobs
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
    return make_video_property(tmp_path / "data" / "properties")


@pytest.fixture
def launched(monkeypatch):
    """Stands in for starting a process: records the command, answers with this process's pid."""
    calls: list[tuple[list[str], Path]] = []

    def fake(command, log_path):
        calls.append((command, log_path))
        return os.getpid()

    monkeypatch.setattr(video_jobs, "_launch", fake)
    return calls


@pytest.fixture
def settings_restored(monkeypatch):
    """--offline rewrites Settings for the process; give them back after the test."""
    for name in (*generate_video._ONLINE_SETTINGS, "image_gen_provider", "tools_enabled",
                 "speech_provider", "video_gen_provider", "mlflow_enabled"):
        monkeypatch.setattr(settings, name, getattr(settings, name), raising=False)


def _wait(condition, seconds=15.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.1)
    return False


# -- starting a plan ------------------------------------------------------------------------------------

def test_a_plan_job_gets_its_record_before_the_process_starts(folder, launched):
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", {"duration_s": 10, "tone": "", "audience": "families",
                                                         "aspect": None})

    record = repository.get_generation(video_id)
    assert record.status == "planning" and record.property_id == "AURELIA-HEIGHTS" and record.pid == os.getpid()
    # None is not an option; an empty string is one (it clears what video_input.json says)
    assert record.request.overrides == {"duration_s": 10, "tone": "", "audience": "families"}
    assert record.request.offline is False and record.request.draft is False

    (command, log_path), = launched
    assert command == [sys.executable, str(video_jobs.JOB_SCRIPT), "plan", video_id]
    assert log_path == repository.video_output_dir(video_id) / "work" / "job.log"
    assert video_jobs.status(video_id).status == "planning"


def test_the_property_folder_is_found_without_regard_to_case(folder, launched):
    video_id = video_jobs.start_plan("aurelia-heights")
    assert repository.get_generation(video_id).property_id == "AURELIA-HEIGHTS"


def test_offline_and_draft_are_kept_on_the_record_and_offline_means_pan_and_zoom(folder, launched):
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", {"model": "auto"}, offline=True, draft=True)
    request = repository.get_generation(video_id).request
    assert (request.offline, request.draft) == (True, True) and request.overrides["model"] == "kenburns"


def test_an_auto_approved_plan_starts_the_full_job_and_keeps_the_prompt_on_the_record(folder, launched):
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", {"duration_s": 30}, auto_approve=True,
                                     prompt="A 30 second reel for families", brief_source="rules",
                                     brief_notes=["Length 30 s from the prompt."])

    request = repository.get_generation(video_id).request
    assert request.auto_approve is True and request.prompt == "A 30 second reel for families"
    assert request.brief_source == "rules" and request.brief_notes == ["Length 30 s from the prompt."]
    assert request.overrides == {"duration_s": 30}
    (command, log_path), = launched
    assert command == [sys.executable, str(video_jobs.JOB_SCRIPT), "full", video_id]
    assert log_path == repository.video_output_dir(video_id) / "work" / "job.log"
    assert video_jobs.status(video_id).status == "planning"


def test_a_plan_with_no_prompt_or_flag_is_the_plan_job_as_before(folder, launched):
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS")
    request = repository.get_generation(video_id).request
    assert (request.auto_approve, request.prompt, request.brief_source, request.brief_notes) == (False, "", "", [])
    assert launched[0][0][-2:] == ["plan", video_id]


def test_a_second_job_is_refused_while_one_runs(folder, launched):
    first = video_jobs.start_plan("AURELIA-HEIGHTS")
    with pytest.raises(JobBusyError, match=first):
        video_jobs.start_plan("AURELIA-HEIGHTS")
    assert len(launched) == 1 and len(repository.list_summaries()) == 1


def test_an_unknown_property_or_a_bad_option_starts_nothing(folder, launched):
    with pytest.raises(JobError, match="No property found"):
        video_jobs.start_plan("NOT-THERE")
    with pytest.raises(JobError):
        video_jobs.start_plan("../etc")
    with pytest.raises(JobError, match="video options are not valid"):
        video_jobs.start_plan("AURELIA-HEIGHTS", {"duration_s": 7})
    assert launched == [] and repository.list_summaries() == []


def test_a_launch_that_fails_gives_the_slot_back(folder, monkeypatch):
    def broken(command, log_path):
        raise OSError("no process for you")

    monkeypatch.setattr(video_jobs, "_launch", broken)
    with pytest.raises(JobError, match="no process for you"):
        video_jobs.start_plan("AURELIA-HEIGHTS")
    (row,) = repository.list_summaries()
    assert row.status == "failed" and "no process for you" in row.error
    monkeypatch.setattr(video_jobs, "_launch", lambda command, log_path: os.getpid())
    video_jobs.start_plan("AURELIA-HEIGHTS")                                  # not stuck behind the dead one


def test_a_full_job_that_cannot_start_fails_the_record_and_gives_the_slot_back(folder, monkeypatch):
    def broken(command, log_path):
        raise OSError("no process for you")

    monkeypatch.setattr(video_jobs, "_launch", broken)
    with pytest.raises(JobError, match="full job could not be started: no process for you"):
        video_jobs.start_plan("AURELIA-HEIGHTS", auto_approve=True)
    (row,) = repository.list_summaries()
    assert row.status == "failed" and "no process for you" in row.error
    monkeypatch.setattr(video_jobs, "_launch", lambda command, log_path: os.getpid())
    video_jobs.start_plan("AURELIA-HEIGHTS", auto_approve=True)


def test_a_job_whose_process_died_is_swept_so_a_new_one_can_start(folder, launched, monkeypatch):
    monkeypatch.setattr(video_jobs, "_launch", lambda command, log_path: 999_999)
    stuck = video_jobs.start_plan("AURELIA-HEIGHTS")
    monkeypatch.setattr(video_jobs, "pid_alive", lambda pid: False)

    fresh = video_jobs.start_plan("AURELIA-HEIGHTS")

    swept = repository.get_generation(stuck)
    assert swept.status == "failed" and "interrupted" in swept.error and "interrupted" in swept.notes
    assert repository.get_generation(fresh).status == "planning"


def test_status_is_read_from_the_database_only(folder, launched, monkeypatch):
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS")
    monkeypatch.setattr(repository, "get_generation", lambda *a, **k: pytest.fail("the record was parsed"))
    summary = video_jobs.status(video_id)
    assert (summary.status, summary.steps_done, summary.current_node) == ("planning", 0, "")
    assert video_jobs.status("vid_unknown0001") is None


# -- the job script, in this process -------------------------------------------------------------------------

def test_a_plan_job_runs_to_planned_and_the_video_can_then_be_produced(folder, launched, settings_restored):
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", offline=True, draft=True)
    assert video_jobs.status(video_id).status == "planning"

    assert run_video_job.main(["plan", video_id]) == 0

    summary = video_jobs.status(video_id)
    assert (summary.status, summary.steps_done, summary.current_node, summary.pid) == ("planned", 5, "", None)
    out = repository.video_output_dir(video_id)
    assert (out / PLAN_STATE_FILE).is_file() and (out / "storyboard.json").is_file()
    with pytest.raises(JobError, match="no video"):
        video_jobs.start_produce("vid_unknown0001")

    video_jobs.start_produce(video_id)
    assert launched[-1][0][-2:] == ["produce", video_id]
    assert video_jobs.status(video_id).status == "producing"
    with pytest.raises(JobError, match="only a planned video"):
        video_jobs.start_produce(video_id)                                    # not twice


def test_a_produce_job_with_no_plan_state_fails_the_record_and_exits_with_one(folder, launched, settings_restored):
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", offline=True, draft=True)
    run_video_job.main(["plan", video_id])
    (repository.video_output_dir(video_id) / PLAN_STATE_FILE).unlink()
    video_jobs.start_produce(video_id)

    assert run_video_job.main(["produce", video_id]) == 1

    record = repository.get_generation(video_id)
    assert record.status == "failed" and PLAN_STATE_FILE in record.error and record.video_path is None


def test_a_plan_job_whose_property_has_gone_fails_the_record(folder, launched, settings_restored):
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", offline=True, draft=True)
    for path in sorted(folder.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    folder.rmdir()

    assert run_video_job.main(["plan", video_id]) == 1

    record = repository.get_generation(video_id)
    assert record.status == "failed" and "No property found" in record.error


def test_a_job_for_an_unknown_video_exits_with_two(settings_restored):
    repository.init_db()
    assert run_video_job.main(["plan", "vid_unknown0001"]) == 2


# -- the full job: plan and produce in one process (the workflow is stubbed: no render) ---------------------------

@pytest.fixture
def full_workflow(monkeypatch):
    """Stands in for run_video_workflow, which would render; records what it was called with."""
    seen: list[tuple[tuple, dict]] = []
    outcome = {"status": "passed", "raises": None}

    def fake(*args, **kwargs):
        seen.append((args, kwargs))
        if outcome["raises"]:
            raise outcome["raises"]
        record = repository.get_generation(kwargs["video_id"])
        record.status = outcome["status"]
        return record

    monkeypatch.setattr("src.workflows.video.run_video_workflow", fake)
    for name in ("run_plan_workflow", "run_produce_workflow"):
        monkeypatch.setattr(f"src.workflows.video.{name}", lambda *a, **k: pytest.fail("not the full job's workflow"))
    return seen, outcome


def test_the_full_job_runs_the_one_process_workflow_with_the_request_from_the_record(folder, launched, full_workflow):
    seen, _ = full_workflow
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", {"duration_s": 10, "tone": "warm"}, draft=True,
                                     auto_approve=True)

    assert run_video_job.main(["full", video_id]) == 0

    (args, kwargs), = seen
    assert Path(args[0]).samefile(folder) and args[1:] == ({"duration_s": 10, "tone": "warm"},)
    assert kwargs == {"offline": False, "draft": True, "refresh_ledger": False, "video_id": video_id}


def test_a_full_job_that_ends_failed_exits_with_one(folder, launched, full_workflow):
    _, outcome = full_workflow
    outcome["status"] = "failed"
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", auto_approve=True)
    assert run_video_job.main(["full", video_id]) == 1


def test_a_full_job_that_breaks_before_the_workflow_can_save_fails_the_record(folder, launched, full_workflow):
    _, outcome = full_workflow
    outcome["raises"] = RuntimeError("the workflow broke")
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", auto_approve=True)

    assert run_video_job.main(["full", video_id]) == 1

    record = repository.get_generation(video_id)
    assert record.status == "failed" and record.error == "the workflow broke"


def test_a_full_job_whose_property_has_gone_fails_the_record(folder, launched, full_workflow):
    seen, _ = full_workflow
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", auto_approve=True)
    for path in sorted(folder.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    folder.rmdir()

    assert run_video_job.main(["full", video_id]) == 1

    assert seen == []
    record = repository.get_generation(video_id)
    assert record.status == "failed" and "No property found" in record.error


# -- producing: the slot ----------------------------------------------------------------------------------

def test_producing_waits_for_the_slot_and_a_failed_launch_puts_the_video_back(folder, launched, monkeypatch,
                                                                              settings_restored):
    first = video_jobs.start_plan("AURELIA-HEIGHTS", offline=True, draft=True)
    run_video_job.main(["plan", first])
    second = video_jobs.start_plan("AURELIA-HEIGHTS", offline=True, draft=True)        # now planning
    with pytest.raises(JobBusyError, match=second):
        video_jobs.start_produce(first)
    assert video_jobs.status(first).status == "planned"

    repository.save_generation(repository.get_generation(second).model_copy(update={"status": "failed"}))
    monkeypatch.setattr(video_jobs, "_launch", lambda command, log_path: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(JobError, match="could not be started"):
        video_jobs.start_produce(first)
    assert video_jobs.status(first).status == "planned"                                # still approvable


# -- cancel -------------------------------------------------------------------------------------------------

def test_cancel_marks_the_job_and_ends_its_process_tree(folder, launched, monkeypatch):
    killed = []
    monkeypatch.setattr(video_jobs, "kill_tree", killed.append)
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS")

    assert video_jobs.cancel(video_id) is True

    assert killed == [os.getpid()]
    summary = video_jobs.status(video_id)
    assert summary.status == "cancelled" and summary.pid is None
    assert video_jobs.cancel(video_id) is False and video_jobs.cancel("vid_unknown0001") is False
    video_jobs.start_plan("AURELIA-HEIGHTS")                                          # the slot is free again


def test_a_planned_video_waiting_for_approval_is_not_a_running_job(folder, launched, monkeypatch, settings_restored):
    monkeypatch.setattr(video_jobs, "kill_tree", lambda pid: pytest.fail("nothing to kill"))
    video_id = video_jobs.start_plan("AURELIA-HEIGHTS", offline=True, draft=True)
    run_video_job.main(["plan", video_id])
    assert video_jobs.cancel(video_id) is False and video_jobs.status(video_id).status == "planned"


# -- real processes -----------------------------------------------------------------------------------------

def test_pid_alive_tells_a_running_process_from_a_finished_one():
    assert video_jobs.pid_alive(os.getpid()) is True
    assert video_jobs.pid_alive(0) is False and video_jobs.pid_alive(-5) is False
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    assert video_jobs.pid_alive(done.pid) is False


def test_a_launched_process_writes_to_its_log_and_outlives_the_call(tmp_path):
    log = tmp_path / "work" / "job.log"
    pid = video_jobs._launch([sys.executable, "-c", "import sys; print('from the job'); print('and this', file=sys.stderr)"],
                             log)
    assert _wait(lambda: not video_jobs.pid_alive(pid))
    text = log.read_text(encoding="utf-8")
    assert "from the job" in text and "and this" in text


def test_cancel_ends_the_children_of_the_job_too(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    code = ("import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            f"open(r'{child_pid_file}', 'w').write(str(child.pid))\n"
            "time.sleep(120)\n")
    pid = video_jobs._launch([sys.executable, "-c", code], tmp_path / "job.log")
    child = None
    try:
        assert _wait(child_pid_file.exists) and _wait(lambda: child_pid_file.read_text().strip() != "")
        child = int(child_pid_file.read_text())
        assert video_jobs.pid_alive(pid) and video_jobs.pid_alive(child)

        video_jobs.kill_tree(pid)

        assert _wait(lambda: not video_jobs.pid_alive(pid)) and _wait(lambda: not video_jobs.pid_alive(child))
    finally:
        for left in (pid, child):
            if left and video_jobs.pid_alive(left):
                video_jobs.kill_tree(left)
