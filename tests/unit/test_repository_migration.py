from __future__ import annotations

import sqlite3

from src.data import repository
from src.data.schemas import AgentStepResult, GenerationRecord, WorkflowEvaluation


def test_init_db_migrates_a_pre_existing_old_schema_database(tmp_path, monkeypatch):
    """Regression test: an app.db created by an earlier version of this
    project (before workflow_evaluation/human_review existed) must not
    break when the app is upgraded. CREATE TABLE IF NOT EXISTS alone does
    NOT add columns to a table that already exists — init_db() must
    explicitly migrate it."""
    db_path = tmp_path / "old.db"

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE generations (
            generation_id TEXT PRIMARY KEY,
            property_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            pdf_path TEXT,
            html_path TEXT,
            screenshot_path TEXT,
            error TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO generations VALUES ('gen_old1','PROP-001','2026-01-01T00:00:00+00:00','completed',NULL,NULL,NULL,NULL)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("src.core.config.settings.database_path", str(db_path), raising=False)

    repository.init_db()  # must not raise

    old_record = repository.get_generation("gen_old1")
    assert old_record is not None
    assert old_record.human_review.status == "pending"  # sensible default for pre-existing rows
    assert old_record.workflow_evaluation is None

    new_record = GenerationRecord(
        generation_id="gen_new1",
        property_id="PROP-001",
        status="completed",
        workflow_evaluation=WorkflowEvaluation(
            steps=[AgentStepResult(node="vision", status="success", latency_ms=42)]
        ),
    )
    repository.save_generation(new_record)
    reloaded = repository.get_generation("gen_new1")
    assert reloaded is not None
    assert reloaded.workflow_evaluation.steps[0].node == "vision"
    assert reloaded.workflow_evaluation.steps[0].latency_ms == 42


def test_blog_db_from_before_editorial_checks_upgrades_and_round_trips(tmp_path, monkeypatch):
    """An existing blog database (no editorial_integrity / issues columns)
    must upgrade in place, and the new fields must survive save + load —
    otherwise earlier versions would reload with editorial_integrity=0."""
    import sqlite3

    from src.core.types import QualityGateResult
    from src.data import blog_repository
    from src.data.schemas import BlogEvaluationResult, BlogGenerationRecord

    db_path = tmp_path / "old_blog.db"
    old = sqlite3.connect(db_path)
    old.executescript("""
        CREATE TABLE blog_generations (generation_id TEXT PRIMARY KEY, blog_id TEXT NOT NULL,
            created_at TEXT NOT NULL, status TEXT NOT NULL, markdown_path TEXT, html_path TEXT,
            error TEXT, workflow_steps_json TEXT, human_review_status TEXT DEFAULT 'pending',
            human_review_comment TEXT DEFAULT '', human_review_reviewed_at TEXT);
        CREATE TABLE blog_evaluations (generation_id TEXT PRIMARY KEY, grounding REAL,
            writing_quality REAL, structural_completeness REAL, quality_score REAL, gate_result TEXT);
    """)
    old.close()
    monkeypatch.setattr("src.core.config.settings.database_path", str(db_path), raising=False)

    blog_repository.init_db()
    record = BlogGenerationRecord(generation_id="G1", blog_id="B1", status="completed")
    record.evaluation = BlogEvaluationResult(
        grounding=80, writing_quality=80, structural_completeness=100, editorial_integrity=60,
        issues=["No call to action."], quality_score=81, gate_result=QualityGateResult.PASS,
    )
    blog_repository.save_generation(record)

    loaded = blog_repository.get_generation("G1")
    assert loaded.evaluation.editorial_integrity == 60
    assert loaded.evaluation.issues == ["No call to action."]
