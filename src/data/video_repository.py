"""
SQLite persistence for video generations: a parallel table to `generations`
and `blog_generations` (see repository.py, blog_repository.py), in the same
data/app.db. The whole VideoGenerationRecord is saved as `record_json`; the
flat columns are for listing and querying. Human review lives in its own
columns and is merged over the record on read.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from pydantic import ValidationError

from src.core.config import settings
from src.core.ids import brief_dir
from src.data.schemas import AgentStepResult, HumanReview, WorkflowEvaluation
from src.data.video_schemas import VideoGenerationRecord

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
_NEW_GENERATION_COLUMNS: dict[str, str] = {}


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


def save_generation(record: VideoGenerationRecord) -> None:
    workflow_steps_json = (
        json.dumps([s.model_dump() for s in record.workflow_evaluation.steps])
        if record.workflow_evaluation
        else None
    )

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO video_generations
                (video_id, property_id, created_at, status, video_path, thumbnail_path,
                 duration_s, aspect, provider, cost_usd, error, workflow_steps_json,
                 human_review_status, human_review_comment, human_review_reviewed_at, record_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                record_json=excluded.record_json
            """,
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
            ),
        )


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
            # was written; those are authoritative.
            record.human_review = human_review
            record.status = row["status"]
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
