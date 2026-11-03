"""
SQLite persistence for blog generations — a parallel table to
`generations` (see repository.py), not a shared one: BlogGenerationRecord
has different fields (markdown_path vs pdf_path, BlogEvaluationResult's
different dimensions) that don't map cleanly onto the brochure schema.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from src.core.config import settings
from src.data.schemas import AgentStepResult, BlogEvaluationResult, BlogGenerationRecord, HumanReview, WorkflowEvaluation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blog_generations (
    generation_id TEXT PRIMARY KEY,
    blog_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    markdown_path TEXT,
    html_path TEXT,
    error TEXT,
    workflow_steps_json TEXT,
    human_review_status TEXT DEFAULT 'pending',
    human_review_comment TEXT DEFAULT '',
    human_review_reviewed_at TEXT
);

CREATE TABLE IF NOT EXISTS blog_evaluations (
    generation_id TEXT PRIMARY KEY REFERENCES blog_generations(generation_id),
    grounding REAL,
    writing_quality REAL,
    structural_completeness REAL,
    quality_score REAL,
    gate_result TEXT
);
"""


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


# Columns added after the first release. Existing databases get them via
# ALTER TABLE on startup (same pattern as repository.py), so upgrading
# never requires deleting data/app.db.
_NEW_EVALUATION_COLUMNS = {
    "editorial_integrity": "REAL DEFAULT 0",
    "issues_json": "TEXT DEFAULT '[]'",
    # The whole BlogEvaluationResult (per-dimension scores, metrics,
    # feedback). The flat columns above stay for old readers.
    "evaluation_json": "TEXT",
}

# The PDF (the primary deliverable) and the full record, so every field
# added to BlogGenerationRecord round-trips without a migration per field.
_NEW_GENERATION_COLUMNS = {
    "pdf_path": "TEXT",
    "record_json": "TEXT",
}


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(_SCHEMA)
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(blog_evaluations)")}
        for column, ddl_type in _NEW_EVALUATION_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE blog_evaluations ADD COLUMN {column} {ddl_type}")
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(blog_generations)")}
        for column, ddl_type in _NEW_GENERATION_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE blog_generations ADD COLUMN {column} {ddl_type}")


def save_generation(record: BlogGenerationRecord) -> None:
    workflow_steps_json = (
        json.dumps([s.model_dump() for s in record.workflow_evaluation.steps])
        if record.workflow_evaluation
        else None
    )

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO blog_generations
                (generation_id, blog_id, created_at, status, markdown_path, html_path,
                 error, workflow_steps_json, human_review_status, human_review_comment,
                 human_review_reviewed_at, pdf_path, record_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(generation_id) DO UPDATE SET
                status=excluded.status,
                markdown_path=excluded.markdown_path,
                html_path=excluded.html_path,
                error=excluded.error,
                workflow_steps_json=excluded.workflow_steps_json,
                pdf_path=excluded.pdf_path,
                record_json=excluded.record_json
            """,
            (
                record.generation_id,
                record.blog_id,
                record.created_at.isoformat(),
                record.status,
                record.markdown_path,
                record.html_path,
                record.error,
                workflow_steps_json,
                record.human_review.status,
                record.human_review.comment,
                record.human_review.reviewed_at.isoformat() if record.human_review.reviewed_at else None,
                record.pdf_path,
                record.model_dump_json(exclude={"human_review"}),
            ),
        )

        if record.evaluation:
            e = record.evaluation
            conn.execute(
                """
                INSERT INTO blog_evaluations
                    (generation_id, grounding, writing_quality, structural_completeness,
                     quality_score, gate_result, editorial_integrity, issues_json, evaluation_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(generation_id) DO UPDATE SET
                    grounding=excluded.grounding,
                    writing_quality=excluded.writing_quality,
                    structural_completeness=excluded.structural_completeness,
                    quality_score=excluded.quality_score,
                    gate_result=excluded.gate_result,
                    editorial_integrity=excluded.editorial_integrity,
                    issues_json=excluded.issues_json,
                    evaluation_json=excluded.evaluation_json
                """,
                (
                    record.generation_id,
                    e.grounding,
                    e.writing_quality,
                    e.structural_completeness,
                    e.quality_score,
                    e.gate_result.value,
                    e.editorial_integrity,
                    json.dumps(e.issues),
                    e.model_dump_json(),
                ),
            )


def update_human_review(generation_id: str, *, status: str, comment: str = "") -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE blog_generations
            SET human_review_status = ?, human_review_comment = ?, human_review_reviewed_at = ?
            WHERE generation_id = ?
            """,
            (status, comment, datetime.now(timezone.utc).isoformat(), generation_id),
        )


def get_generation(generation_id: str) -> Optional[BlogGenerationRecord]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM blog_generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
        if not row:
            return None

        eval_row = conn.execute(
            "SELECT * FROM blog_evaluations WHERE generation_id = ?", (generation_id,)
        ).fetchone()

        evaluation = None
        if eval_row and "evaluation_json" in eval_row.keys() and eval_row["evaluation_json"]:
            evaluation = BlogEvaluationResult.model_validate_json(eval_row["evaluation_json"])
        elif eval_row:
            evaluation = BlogEvaluationResult(
                grounding=eval_row["grounding"],
                writing_quality=eval_row["writing_quality"],
                structural_completeness=eval_row["structural_completeness"],
                editorial_integrity=eval_row["editorial_integrity"] or 0,
                issues=json.loads(eval_row["issues_json"] or "[]"),
                quality_score=eval_row["quality_score"],
                gate_result=eval_row["gate_result"],
            )

        workflow_evaluation = None
        if row["workflow_steps_json"]:
            steps = [AgentStepResult(**s) for s in json.loads(row["workflow_steps_json"])]
            workflow_evaluation = WorkflowEvaluation(steps=steps)

        human_review = HumanReview(
            status=row["human_review_status"] or "pending",
            comment=row["human_review_comment"] or "",
            reviewed_at=row["human_review_reviewed_at"] or None,
        )

        if "record_json" in row.keys() and row["record_json"]:
            record = BlogGenerationRecord.model_validate_json(row["record_json"])
            # Human review and evaluation are updated in their own columns
            # after the record was written; those are authoritative.
            record.human_review = human_review
            record.evaluation = evaluation or record.evaluation
            record.status = row["status"]
            return record

        return BlogGenerationRecord(
            generation_id=row["generation_id"],
            blog_id=row["blog_id"],
            created_at=row["created_at"],
            status=row["status"],
            markdown_path=row["markdown_path"],
            html_path=row["html_path"],
            pdf_path=row["pdf_path"] if "pdf_path" in row.keys() else None,
            error=row["error"],
            workflow_evaluation=workflow_evaluation,
            evaluation=evaluation,
            human_review=human_review,
        )


def list_generations(blog_id: Optional[str] = None) -> list[BlogGenerationRecord]:
    with get_connection() as conn:
        if blog_id:
            rows = conn.execute(
                "SELECT generation_id FROM blog_generations WHERE blog_id = ? ORDER BY created_at DESC",
                (blog_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT generation_id FROM blog_generations ORDER BY created_at DESC"
            ).fetchall()
    return [get_generation(r["generation_id"]) for r in rows]  # type: ignore[misc]
