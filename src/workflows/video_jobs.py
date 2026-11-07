"""
Background video jobs: a video run as a separate process, in two jobs.

    start_plan(property_id, overrides)   planning -> planned (waiting for approval)
    start_produce(video_id)              planned  -> producing -> passed / warned / failed

or, when no one reviews the plan (start_plan(..., auto_approve=True)), in one:

    full job                             planning -> producing -> passed / warned / failed

Each job is `python scripts/run_video_job.py plan|produce|full <video_id>`, started
with the running interpreter, in the project folder, with its output going to
<video folder>/work/job.log. A process of its own, not a thread, because a run
changes the process-wide Settings (workflows/video.py wraps it in the run
policy, and --offline blanks the keys), because ffmpeg and Chromium need the
memory to themselves, and because it must outlive a Streamlit rerun or
restart. The job saves its state in the database after every node
(video_repository.update_progress); a page or an API call reads it back with
status() and never talks to the process.

One job at a time: begin_job / begin_produce check and claim the slot in a
single SQLite transaction, so two callers (Streamlit and uvicorn) cannot both
start one. A job whose process died is marked failed ("interrupted") by
sweep(), which every start runs first.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from src.agents.video import brief as brief_agent
from src.core.config import PROJECT_ROOT, settings
from src.core.exceptions import IngestionError, JobError
from src.core.ids import brief_dir
from src.data import video_repository as repository
from src.data.video_schemas import STATUS_PLANNED, STATUS_PLANNING, VideoGenerationRecord, VideoRequest, VideoSummary

logger = logging.getLogger(__name__)

JOB_SCRIPT = PROJECT_ROOT / "scripts" / "run_video_job.py"
LOG_NAME = "job.log"
MODES = ("plan", "produce", "full")


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

def pid_alive(pid: int) -> bool:
    """Whether a process with this pid is running. (os.kill(pid, 0) would
    terminate it on Windows, so that platform asks the system instead.) A pid
    can be reused by another program after a crash; the sweep then leaves the
    row alone until that program ends."""
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)           # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                   # access denied: it exists
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_tree(pid: int) -> None:
    """Ends the process and everything it started (ffmpeg, Chromium)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)                 # the job leads its own group (_launch)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def _launch(command: list[str], log_path: Path) -> int:
    """Starts `command` detached from this process's console and returns its pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    options: dict = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    with open(log_path, "ab") as log:
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, env=env, **options)
    return process.pid


def _start(mode: str, video_id: str) -> int:
    out_dir = repository.video_output_dir(video_id)
    return _launch([sys.executable, str(JOB_SCRIPT), mode, video_id], out_dir / "work" / LOG_NAME)


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def sweep() -> list[str]:
    """Marks every planning/producing job whose process has gone as failed
    ("interrupted"); returns their ids."""
    repository.init_db()
    swept = repository.sweep_interrupted(pid_alive)
    for video_id in swept:
        logger.warning("Video %s was interrupted: its process is gone", video_id)
    return swept


def resolve_property_dir(property_id: str) -> Path:
    """data/properties/<property_id>; the folder name is matched without regard to case.
    Raises JobError for an id that is not a folder name or has no folder."""
    properties = settings.data_dir_resolved / "properties"
    try:
        folder = brief_dir(properties, property_id)
    except ValueError as exc:
        raise JobError(str(exc)) from exc
    if not folder.exists() and properties.is_dir():
        folder = next((d for d in sorted(properties.iterdir())
                       if d.is_dir() and d.name.lower() == property_id.strip().lower()), folder)
    if not folder.is_dir():
        raise JobError(f"No property found at {folder}. Add it with the form or scripts/ingest_property.py first.")
    return folder


def start_plan(property_id: str, overrides: Optional[dict] = None, *, offline: bool = False, draft: bool = False,
               refresh_ledger: bool = False, auto_approve: bool = False, prompt: str = "", brief_source: str = "",
               brief_notes: Optional[list[str]] = None, sources: Optional[dict[str, str]] = None,
               recipe_id: str = "") -> str:
    """Creates the record and starts the plan job; returns the video id at once.
    With `auto_approve` the job is the full one: it plans and then produces
    without stopping for approval. `prompt`, `brief_source` and `brief_notes`
    are kept on the record's request (the prompt box), with `sources` (where each option came from). Raises JobBusyError when
    another video is planning or producing, JobError for an unknown property
    or options that are not valid."""
    sweep()
    folder = resolve_property_dir(property_id)
    given = {k: v for k, v in (overrides or {}).items() if v is not None}
    if offline:
        given["model"] = "kenburns"
    try:
        brief_agent.load_video_input(folder, given)               # the options are checked before anything is made
    except IngestionError as exc:
        raise JobError(str(exc)) from exc

    video_id = f"vid_{uuid.uuid4().hex[:10]}"
    record = VideoGenerationRecord(
        video_id=video_id, property_id=folder.name, status=STATUS_PLANNING,
        request=VideoRequest(property_id=folder.name, overrides=given, offline=offline, draft=draft,
                             refresh_ledger=refresh_ledger, auto_approve=auto_approve, prompt=prompt,
                             brief_source=brief_source, brief_notes=list(brief_notes or []),
                             sources=dict(sources or {}), recipe_id=recipe_id))
    (repository.video_output_dir(video_id) / "work").mkdir(parents=True, exist_ok=True)
    repository.begin_job(record)
    _launch_or_fail(record, "full" if auto_approve else "plan")
    return video_id


def start_produce(video_id: str) -> None:
    """Starts the produce job of a planned video (the person has approved the
    storyboard). Raises JobError when the video is not planned and JobBusyError
    when another is running."""
    sweep()
    record = repository.begin_produce(video_id)
    _launch_or_fail(record, "produce")


def _launch_or_fail(record: VideoGenerationRecord, mode: str) -> None:
    try:
        pid = _start(mode, record.video_id)
    except Exception as exc:  # noqa: BLE001 - whatever stopped the launch, the slot must be given back
        logger.exception("Could not start the %s job of %s", mode, record.video_id)
        if mode == "produce":                                       # the plan is still there to approve again
            record.status, record.error = STATUS_PLANNED, f"The job could not be started: {exc}"
        else:
            record.status, record.error = "failed", f"The job could not be started: {exc}"
        repository.save_generation(record)
        raise JobError(f"The {mode} job could not be started: {exc}") from exc
    repository.set_pid(record.video_id, pid)


def status(video_id: str) -> Optional[VideoSummary]:
    """Where the job is: read from the database only (None for an unknown id)."""
    return repository.get_summary(video_id)


def cancel(video_id: str) -> bool:
    """Stops a planning or producing job: marks it cancelled, then ends its
    process and the programs it started. False when it was not running."""
    summary = repository.get_summary(video_id)
    if summary is None or summary.status not in repository.ACTIVE_STATUSES:
        return False
    if not repository.mark_cancelled(video_id):
        return False
    if summary.pid:
        kill_tree(summary.pid)
    return True
