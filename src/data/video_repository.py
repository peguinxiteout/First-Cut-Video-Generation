"""
SQLite persistence for video generations: a parallel table to `generations`
and `blog_generations` (see repository.py, blog_repository.py), in the same
data/app.db. The whole VideoGenerationRecord is saved as `record_json`; the
flat columns are for listing and querying. Human review lives in its own
columns and is merged over the record on read.

A background job also keeps its state in flat columns (status, current_node,
steps_done, updated_at, pid) so a poll reads a few bytes and a node finishing
writes a few bytes: update_progress touches those and workflow_steps_json,
never record_json. begin_job and begin_produce check "is anything running"
and claim the slot in one BEGIN IMMEDIATE transaction, so two processes
(Streamlit, uvicorn) cannot both start a job.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

from pydantic import ValidationError

from src.core.config import settings
from src.core.exceptions import JobBusyError, JobError
from src.core.ids import brief_dir
from src.data.schemas import AgentStepResult, HumanReview, WorkflowEvaluation
from src.data.video_schemas import (
    ACTIVE_STATUSES, STATUS_CANCELLED, STATUS_PLANNED, STATUS_PRODUCING, VideoGenerationRecord, VideoSummary,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS video_generations (
    video_id TEXT PRIMARY KEY,
    property_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    video_path TEXT,
    thumbnail_path TEXT,
    duration_s INTEGER,
    aspect TEXT,
    provider TEXT,
    cost_usd REAL DEFAULT 0,
    error TEXT,
    workflow_steps_json TEXT,
    human_review_status TEXT DEFAULT 'pending',
    human_review_comment TEXT DEFAULT '',
    human_review_reviewed_at TEXT,
    record_json TEXT
);
"""

# Columns added after the first release get an ALTER TABLE in init_db(), as
# in repository.py and blog_repository.py, so an old data/app.db keeps working.
_NEW_GENERATION_COLUMNS: dict[str, str] = {
    "current_node": "TEXT DEFAULT ''",
    "steps_done": "INTEGER DEFAULT 0",
    "updated_at": "TEXT",
    "pid": "INTEGER",
    "request_json": "TEXT",
    "gate_result": "TEXT DEFAULT ''",
    "quality_score": "REAL",
    "started_at": "TEXT",
}

# Rows a job may still be working on. "running" is what a run made before the
# job statuses existed left behind.
_ACTIVE = (*ACTIVE_STATUSES, "running")
_ACTIVE_SQL = ", ".join(f"'{s}'" for s in _ACTIVE)
# A row with no pid is taken for dead when nothing has touched it for this long.
STALE_AFTER_S = 30 * 60


def video_output_dir(video_id: str) -> Path:
    """data/generated_videos/<video_id>, worked out when called (not at
    import, so a changed DATA_DIR is honoured). Raises ValueError for an id
    that is not a single folder name."""
    return brief_dir(settings.data_dir_resolved / "generated_videos", video_id)


def _db_path() -> Path:
    path = settings.database_path_resolved
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(_SCHEMA)
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(video_generations)")}
        for column, ddl_type in _NEW_GENERATION_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE video_generations ADD COLUMN {column} {ddl_type}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(value: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def _col(row: sqlite3.Row, name: str, default=None):
    """A column added after the first release; a database read before init_db has none."""
    return row[name] if name in row.keys() and row[name] is not None else default


# ---------------------------------------------------------------------------
# Save and read
# ---------------------------------------------------------------------------

def _save(conn: sqlite3.Connection, record: VideoGenerationRecord) -> None:
    workflow_steps_json = (
        json.dumps([s.model_dump() for s in record.workflow_evaluation.steps])
        if record.workflow_evaluation
        else None
    )
    gate = record.evaluation.gate_result.value if record.evaluation else ""
    score = record.evaluation.quality_score if record.evaluation else None
    conn.execute(
        """
        INSERT INTO video_generations
            (video_id, property_id, created_at, status, video_path, thumbnail_path,
             duration_s, aspect, provider, cost_usd, error, workflow_steps_json,
             human_review_status, human_review_comment, human_review_reviewed_at, record_json,
             current_node, steps_done, updated_at, pid, request_json, gate_result, quality_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            status=excluded.status,
            video_path=excluded.video_path,
            thumbnail_path=excluded.thumbnail_path,
            duration_s=excluded.duration_s,
            aspect=excluded.aspect,
            provider=excluded.provider,
            cost_usd=excluded.cost_usd,
            error=excluded.error,
            workflow_steps_json=excluded.workflow_steps_json,
            record_json=excluded.record_json,
            current_node=excluded.current_node,
            steps_done=excluded.steps_done,
            updated_at=excluded.updated_at,
            request_json=excluded.request_json,
            gate_result=excluded.gate_result,
            quality_score=excluded.quality_score,
            pid=CASE WHEN excluded.status IN (%s) THEN COALESCE(excluded.pid, video_generations.pid)
                     ELSE NULL END
        WHERE video_generations.status != 'cancelled'
        """ % _ACTIVE_SQL,
        (
            record.video_id,
            record.property_id,
            record.created_at.isoformat(),
            record.status,
            record.video_path,
            record.thumbnail_path,
            record.duration_s,
            record.aspect,
            record.provider,
            record.cost_usd,
            record.error,
            workflow_steps_json,
            record.human_review.status,
            record.human_review.comment,
            record.human_review.reviewed_at.isoformat() if record.human_review.reviewed_at else None,
            record.model_dump_json(exclude={"human_review"}),
            record.current_node,
            record.steps_done,
            _now(),
            record.pid,
            record.request.model_dump_json() if record.request else None,
            gate,
            score,
        ),
    )


def save_generation(record: VideoGenerationRecord) -> None:
    with get_connection() as conn:
        _save(conn, record)


def update_human_review(video_id: str, *, status: str, comment: str = "") -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE video_generations
            SET human_review_status = ?, human_review_comment = ?, human_review_reviewed_at = ?
            WHERE video_id = ?
            """,
            (status, comment, datetime.now(timezone.utc).isoformat(), video_id),
        )


def _from_row(row: sqlite3.Row) -> VideoGenerationRecord:
    workflow_evaluation = None
    if row["workflow_steps_json"]:
        steps = [AgentStepResult(**s) for s in json.loads(row["workflow_steps_json"])]
        workflow_evaluation = WorkflowEvaluation(steps=steps)

    human_review = HumanReview(
        status=row["human_review_status"] or "pending",
        comment=row["human_review_comment"] or "",
        reviewed_at=row["human_review_reviewed_at"] or None,
    )

    if row["record_json"]:
        try:
            record = VideoGenerationRecord.model_validate_json(row["record_json"])
        except ValidationError as exc:
            # A record written by an older schema: fall through to the columns.
            logger.warning("record_json for %s no longer validates (%s); reading the flat columns",
                           row["video_id"], " ".join(str(exc).split()))
        else:
            # Human review is updated in its own columns after the record
            # was written; those are authoritative. So are the job columns,
            # which a node finishing updates without rewriting record_json.
            record.human_review = human_review
            record.status = row["status"]
            record.error = row["error"]
            record.current_node = _col(row, "current_node", "")
            record.steps_done = _col(row, "steps_done", 0)
            record.updated_at = _parse(_col(row, "updated_at"))
            record.pid = _col(row, "pid")
            if workflow_evaluation is not None:
                record.workflow_evaluation = workflow_evaluation
            return record

    return VideoGenerationRecord(
        video_id=row["video_id"],
        property_id=row["property_id"],
        created_at=row["created_at"],
        status=row["status"],
        video_path=row["video_path"],
        thumbnail_path=row["thumbnail_path"],
        duration_s=row["duration_s"] or 15,
        aspect=row["aspect"] or "9:16",
        provider=row["provider"] or "auto",
        cost_usd=row["cost_usd"] or 0.0,
        error=row["error"],
        current_node=_col(row, "current_node", ""),
        steps_done=_col(row, "steps_done", 0),
        updated_at=_parse(_col(row, "updated_at")),
        pid=_col(row, "pid"),
        workflow_evaluation=workflow_evaluation,
        human_review=human_review,
    )


def get_generation(video_id: str) -> Optional[VideoGenerationRecord]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM video_generations WHERE video_id = ?", (video_id,)).fetchone()
    return _from_row(row) if row else None


def list_generations(property_id: Optional[str] = None) -> list[VideoGenerationRecord]:
    with get_connection() as conn:
        if property_id:
            rows = conn.execute(
                "SELECT * FROM video_generations WHERE property_id = ? ORDER BY created_at DESC",
                (property_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM video_generations ORDER BY created_at DESC").fetchall()
    return [_from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# History: the flat columns only
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS = ("video_id, property_id, created_at, updated_at, started_at, status, current_node, steps_done, duration_s, "
                    "aspect, provider, cost_usd, gate_result, quality_score, video_path, thumbnail_path, error, "
                    "pid, human_review_status")


def _summary(row: sqlite3.Row) -> VideoSummary:
    return VideoSummary(
        video_id=row["video_id"], property_id=row["property_id"], created_at=row["created_at"],
        updated_at=_parse(_col(row, "updated_at")), started_at=_parse(_col(row, "started_at")), status=row["status"],
        current_node=_col(row, "current_node", ""), steps_done=_col(row, "steps_done", 0),
        duration_s=row["duration_s"] or 15, aspect=row["aspect"] or "9:16", provider=row["provider"] or "auto",
        cost_usd=row["cost_usd"] or 0.0, gate_result=_col(row, "gate_result", ""),
        quality_score=_col(row, "quality_score"), video_path=row["video_path"],
        thumbnail_path=row["thumbnail_path"], error=row["error"], pid=_col(row, "pid"),
        human_review_status=row["human_review_status"] or "pending")


def list_summaries(*, limit: int = 50, offset: int = 0, statuses: Optional[Sequence[str]] = None,
                   property_id: Optional[str] = None) -> list[VideoSummary]:
    """The history, newest first, read from the flat columns (record_json is
    not parsed). `statuses` keeps only those statuses."""
    where, args = [], []
    if statuses:
        where.append(f"status IN ({', '.join('?' * len(statuses))})")
        args += list(statuses)
    if property_id:
        where.append("property_id = ?")
        args.append(property_id)
    sql = f"SELECT {_SUMMARY_COLUMNS} FROM video_generations"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    with get_connection() as conn:
        rows = conn.execute(sql, (*args, max(0, limit), max(0, offset))).fetchall()
    return [_summary(r) for r in rows]


def list_by_status(*statuses: str, limit: int = 50, offset: int = 0) -> list[VideoSummary]:
    return list_summaries(limit=limit, offset=offset, statuses=statuses)


def get_summary(video_id: str) -> Optional[VideoSummary]:
    with get_connection() as conn:
        row = conn.execute(f"SELECT {_SUMMARY_COLUMNS} FROM video_generations WHERE video_id = ?",
                           (video_id,)).fetchone()
    return _summary(row) if row else None


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@contextmanager
def _exclusive() -> Iterator[sqlite3.Connection]:
    """A connection holding SQLite's write lock from the first statement, so a
    check and the write that depends on it cannot be split by another process."""
    conn = sqlite3.connect(_db_path(), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()


def _busy(conn: sqlite3.Connection, except_id: str = "") -> Optional[sqlite3.Row]:
    return conn.execute(
        f"SELECT video_id, status FROM video_generations WHERE status IN ({_ACTIVE_SQL}) AND video_id != ? LIMIT 1",
        (except_id,)).fetchone()


def _busy_error(row: sqlite3.Row) -> JobBusyError:
    return JobBusyError(f"Video {row['video_id']} is still {row['status']}. One video is made at a time; "
                        "wait for it to finish or cancel it.")


def begin_job(record: VideoGenerationRecord) -> None:
    """Saves a new record unless a job is planning or producing (JobBusyError).
    The check and the insert are one transaction."""
    with _exclusive() as conn:
        busy = _busy(conn)
        if busy:
            raise _busy_error(busy)
        _save(conn, record)
        conn.execute("UPDATE video_generations SET started_at = ? WHERE video_id = ?", (_now(), record.video_id))


def begin_produce(video_id: str) -> VideoGenerationRecord:
    """Moves a planned record to producing and returns it, unless a job is
    running (JobBusyError) or the record is not waiting for approval (JobError)."""
    with _exclusive() as conn:
        row = conn.execute("SELECT status FROM video_generations WHERE video_id = ?", (video_id,)).fetchone()
        if row is None:
            raise JobError(f"There is no video {video_id}.")
        if row["status"] != STATUS_PLANNED:
            raise JobError(f"Video {video_id} is '{row['status']}'; only a planned video can be produced.")
        busy = _busy(conn, video_id)
        if busy:
            raise _busy_error(busy)
        now = _now()
        conn.execute("UPDATE video_generations SET status = ?, current_node = '', updated_at = ?, started_at = ?, "
                     "error = NULL WHERE video_id = ?", (STATUS_PRODUCING, now, now, video_id))
    return get_generation(video_id)


def update_progress(video_id: str, *, status: Optional[str] = None, current_node: Optional[str] = None,
                    steps: Optional[Sequence[AgentStepResult]] = None) -> bool:
    """The cheap write after each node: status, current node, the steps so far
    and a timestamp, in their own columns. Does nothing (False) for a row that
    is no longer active, so a job cannot undo a cancel."""
    sets, args = ["updated_at = ?"], [_now()]
    if status is not None:
        sets.append("status = ?")
        args.append(status)
    if current_node is not None:
        sets.append("current_node = ?")
        args.append(current_node)
    if steps is not None:
        sets += ["workflow_steps_json = ?", "steps_done = ?"]
        args += [json.dumps([s.model_dump() for s in steps]), sum(1 for s in steps if s.status == "success")]
    with get_connection() as conn:
        cursor = conn.execute(
            f"UPDATE video_generations SET {', '.join(sets)} WHERE video_id = ? AND status IN ({_ACTIVE_SQL})",
            (*args, video_id))
        return cursor.rowcount > 0


def set_pid(video_id: str, pid: Optional[int]) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE video_generations SET pid = ?, updated_at = ? WHERE video_id = ?",
                     (pid, _now(), video_id))


def mark_cancelled(video_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.execute(
            f"UPDATE video_generations SET status = ?, error = 'Cancelled.', pid = NULL, updated_at = ? "
            f"WHERE video_id = ? AND status IN ({_ACTIVE_SQL})", (STATUS_CANCELLED, _now(), video_id))
        return cursor.rowcount > 0


def sweep_interrupted(is_alive: Callable[[int], bool], *, stale_after_s: float = STALE_AFTER_S) -> list[str]:
    """Marks planning/producing rows whose process is gone as failed
    ("interrupted"). A row with no pid is taken for dead only when nothing has
    touched it for `stale_after_s`. Returns the ids it changed."""
    changed = []
    now = datetime.now(timezone.utc)
    with get_connection() as conn:
        rows = conn.execute(f"SELECT video_id, pid, updated_at, created_at FROM video_generations "
                            f"WHERE status IN ({_ACTIVE_SQL})").fetchall()
    for row in rows:
        if row["pid"]:
            dead = not is_alive(row["pid"])
        else:
            touched = _parse(row["updated_at"]) or _parse(row["created_at"]) or now
            dead = (now - touched).total_seconds() > stale_after_s
        if not dead:
            continue
        record = get_generation(row["video_id"])
        if record is None or record.status not in _ACTIVE:
            continue
        record.status, record.error = "failed", "interrupted: the job's process is gone before it finished"
        record.notes = list(dict.fromkeys([*record.notes, "interrupted"]))
        save_generation(record)
        changed.append(row["video_id"])
    return changed
