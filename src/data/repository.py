"""
Minimal SQLite persistence layer. Deliberately raw sqlite3 (no ORM) to keep
the POC dependency-light. The interface is narrow enough that swapping in
PostgreSQL later just means rewriting this one file.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from src.core.config import settings
from src.data.schemas import (
    AgentStepResult,
    EvaluationResult,
    GenerationRecord,
    HumanReview,
    WorkflowEvaluation,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    generation_id TEXT PRIMARY KEY,
    property_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    pdf_path TEXT,
    html_path TEXT,
    screenshot_path TEXT,
    error TEXT,
    workflow_steps_json TEXT,
    human_review_status TEXT DEFAULT 'pending',
    human_review_comment TEXT DEFAULT '',
    human_review_reviewed_at TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    generation_id TEXT PRIMARY KEY REFERENCES generations(generation_id),
    factual_consistency REAL,
    visual_design_quality REAL,
    content_quality REAL,
    completeness REAL,
    quality_score REAL,
    gate_result TEXT
);
"""

# Columns added to `generations`/`evaluations` after their initial release.
# Used by init_db() to migrate an app.db created by an earlier version of
# this project — see the comment there. Older evaluation dimension columns
# (visual_quality, layout_quality, brand_consistency, factual_accuracy,
# technical_validity, issues_json, unsupported_claims_json) are simply left
# in place, unused, on existing databases rather than dropped — SQLite
# makes dropping columns unnecessarily invasive for a POC-scale app, and
# old rows keep reading back fine either way.
_NEW_GENERATION_COLUMNS = {
    "workflow_steps_json": "TEXT",
    "human_review_status": "TEXT DEFAULT 'pending'",
    "human_review_comment": "TEXT DEFAULT ''",
    "human_review_reviewed_at": "TEXT",
    # The whole record as JSON, so fields added to GenerationRecord since
    # (template, page count, agent reports, measured checks) survive a
    # reload without a column each. The columns above stay for querying.
    "record_json": "TEXT",
}
_NEW_EVALUATION_COLUMNS = {
    "factual_consistency": "REAL",
    "visual_design_quality": "REAL",
    "layout_integrity": "REAL",          # NULL = not measured
    "issues_json": "TEXT DEFAULT '[]'",
}


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
        existing_gen_columns = {row["name"] for row in conn.execute("PRAGMA table_info(generations)")}
        for column, ddl_type in _NEW_GENERATION_COLUMNS.items():
            if column not in existing_gen_columns:
                conn.execute(f"ALTER TABLE generations ADD COLUMN {column} {ddl_type}")

        existing_eval_columns = {row["name"] for row in conn.execute("PRAGMA table_info(evaluations)")}
        for column, ddl_type in _NEW_EVALUATION_COLUMNS.items():
            if column not in existing_eval_columns:
                conn.execute(f"ALTER TABLE evaluations ADD COLUMN {column} {ddl_type}")


def save_generation(record: GenerationRecord) -> None:
    workflow_steps_json = (
        json.dumps([s.model_dump() for s in record.workflow_evaluation.steps])
        if record.workflow_evaluation
        else None
    )

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO generations
                (generation_id, property_id, created_at, status, pdf_path, html_path,
                 screenshot_path, error, workflow_steps_json,
                 human_review_status, human_review_comment, human_review_reviewed_at, record_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(generation_id) DO UPDATE SET
                record_json=excluded.record_json,
                status=excluded.status,
                pdf_path=excluded.pdf_path,
                html_path=excluded.html_path,
                screenshot_path=excluded.screenshot_path,
                error=excluded.error,
                workflow_steps_json=excluded.workflow_steps_json
            """,
            (
                record.generation_id,
                record.property_id,
                record.created_at.isoformat(),
                record.status,
                record.pdf_path,
                record.html_path,
                record.screenshot_path,
                record.error,
                workflow_steps_json,
                record.human_review.status,
                record.human_review.comment,
                record.human_review.reviewed_at.isoformat() if record.human_review.reviewed_at else None,
                record.model_dump_json(),
            ),
        )

        if record.evaluation:
            e = record.evaluation
            conn.execute(
                """
                INSERT INTO evaluations
                    (generation_id, factual_consistency, visual_design_quality,
                     content_quality, quality_score, gate_result, layout_integrity, issues_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(generation_id) DO UPDATE SET
                    factual_consistency=excluded.factual_consistency,
                    visual_design_quality=excluded.visual_design_quality,
                    content_quality=excluded.content_quality,
                    quality_score=excluded.quality_score,
                    gate_result=excluded.gate_result,
                    layout_integrity=excluded.layout_integrity,
                    issues_json=excluded.issues_json
                """,
                (
                    record.generation_id,
                    e.factual_consistency,
                    e.visual_design_quality,
                    e.content_quality,
                    e.quality_score,
                    e.gate_result.value,
                    e.layout_integrity,
                    json.dumps(e.issues),
                ),
            )


def update_human_review(generation_id: str, *, status: str, comment: str = "") -> None:
    """Layer 3 (Human Review): records a person's approve/request-revision
    decision against an existing generation. Called from the Streamlit UI."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE generations
            SET human_review_status = ?, human_review_comment = ?, human_review_reviewed_at = ?
            WHERE generation_id = ?
            """,
            (status, comment, datetime.now(timezone.utc).isoformat(), generation_id),
        )


def get_generation(generation_id: str) -> Optional[GenerationRecord]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
        if not row:
            return None

        eval_row = conn.execute(
            "SELECT * FROM evaluations WHERE generation_id = ?", (generation_id,)
        ).fetchone()

        evaluation = None
        if eval_row:
            eval_keys = eval_row.keys()
            evaluation = EvaluationResult(
                factual_consistency=eval_row["factual_consistency"] if "factual_consistency" in eval_keys and eval_row["factual_consistency"] is not None else 0,
                visual_design_quality=eval_row["visual_design_quality"] if "visual_design_quality" in eval_keys and eval_row["visual_design_quality"] is not None else 0,
                content_quality=eval_row["content_quality"] or 0,
                layout_integrity=eval_row["layout_integrity"] if "layout_integrity" in eval_keys else None,
                issues=json.loads(eval_row["issues_json"] or "[]") if "issues_json" in eval_keys else [],
                quality_score=eval_row["quality_score"] or 0,
                gate_result=eval_row["gate_result"] or "fail",
            )

        row_keys = row.keys()
        human_review = HumanReview(
            status=row["human_review_status"] if "human_review_status" in row_keys and row["human_review_status"] else "pending",
            comment=row["human_review_comment"] if "human_review_comment" in row_keys and row["human_review_comment"] else "",
            reviewed_at=(
                row["human_review_reviewed_at"]
                if "human_review_reviewed_at" in row_keys and row["human_review_reviewed_at"]
                else None
            ),
        )
        if "record_json" in row_keys and row["record_json"]:
            try:
                stored = GenerationRecord.model_validate_json(row["record_json"])
                stored.human_review = human_review
                return stored
            except ValueError:
                pass          # an older shape: rebuild from the columns below

        workflow_evaluation = None
        if "workflow_steps_json" in row_keys and row["workflow_steps_json"]:
            steps = [AgentStepResult(**s) for s in json.loads(row["workflow_steps_json"])]
            workflow_evaluation = WorkflowEvaluation(steps=steps)

        return GenerationRecord(
            generation_id=row["generation_id"],
            property_id=row["property_id"],
            created_at=row["created_at"],
            status=row["status"],
            pdf_path=row["pdf_path"],
            html_path=row["html_path"],
            screenshot_path=row["screenshot_path"],
            error=row["error"],
            workflow_evaluation=workflow_evaluation,
            evaluation=evaluation,
            human_review=human_review,
        )


def list_generations(property_id: Optional[str] = None) -> list[GenerationRecord]:
    with get_connection() as conn:
        if property_id:
            rows = conn.execute(
                "SELECT generation_id FROM generations WHERE property_id = ? ORDER BY created_at DESC",
                (property_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT generation_id FROM generations ORDER BY created_at DESC"
            ).fetchall()
    return [get_generation(r["generation_id"]) for r in rows]  # type: ignore[misc]
