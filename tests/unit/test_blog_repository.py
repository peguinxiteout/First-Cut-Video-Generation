"""Blog records persist every field the UI shows — including the PDF, the
per-dimension evaluation and the revision history — and databases created
before those existed are migrated in place."""
from __future__ import annotations

import sqlite3

from src.core.types import QualityGateResult
from src.data import blog_repository as repo
from src.data.schemas import (
    AgentStepResult, BlogDimensionScore, BlogEvaluationResult, BlogGenerationRecord, BlogRevisionRound,
    BlogValidationIssue, BlogValidationReport, WorkflowEvaluation,
)


def test_full_record_round_trips_and_human_review_stays_authoritative(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "r.db"), raising=False)
    repo.init_db()
    record = BlogGenerationRecord(
        generation_id="g1", blog_id="b1", status="completed", pdf_path="/x/article.pdf", markdown_path="/x/c.md",
        title="T", archetype="comparison", pdf_pages=5, writer_mode="llm",
        evaluation=BlogEvaluationResult(quality_score=81.5, gate_result=QualityGateResult.PASS,
                                        dimensions=[BlogDimensionScore(name="accuracy", score=90, threshold=80)],
                                        metrics={"word_count": 1200.0}, feedback=["x"]),
        workflow_evaluation=WorkflowEvaluation(steps=[AgentStepResult(node="writer", latency_ms=5)]),
        revision_history=[BlogRevisionRound(round=0, quality_score=70, decision="revise")],
        validation=BlogValidationReport(fixes=["Split bullets"], issues=[BlogValidationIssue(code="c", message="m")]))
    repo.save_generation(record)
    repo.update_human_review("g1", status="approved", comment="Looks good")

    back = repo.get_generation("g1")
    assert back.pdf_path == "/x/article.pdf" and back.pdf_pages == 5 and back.archetype == "comparison"
    assert back.evaluation.dimensions[0].name == "accuracy" and back.evaluation.metrics["word_count"] == 1200.0
    assert back.revision_history[0].decision == "revise"
    assert back.validation.fixes == ["Split bullets"]
    assert back.human_review.status == "approved" and back.human_review.comment == "Looks good"
    assert [g.generation_id for g in repo.list_generations("b1")] == ["g1"]


def test_an_old_database_is_migrated_and_old_rows_still_load(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE blog_generations (generation_id TEXT PRIMARY KEY, blog_id TEXT NOT NULL,
            created_at TEXT NOT NULL, status TEXT NOT NULL, markdown_path TEXT, html_path TEXT, error TEXT,
            workflow_steps_json TEXT, human_review_status TEXT DEFAULT 'pending',
            human_review_comment TEXT DEFAULT '', human_review_reviewed_at TEXT);
        CREATE TABLE blog_evaluations (generation_id TEXT PRIMARY KEY, grounding REAL, writing_quality REAL,
            structural_completeness REAL, quality_score REAL, gate_result TEXT);
        INSERT INTO blog_generations (generation_id, blog_id, created_at, status, markdown_path)
            VALUES ('old', 'b', '2026-01-01T00:00:00+00:00', 'completed', '/old.md');
        INSERT INTO blog_evaluations VALUES ('old', 80, 70, 100, 77.5, 'pass');
    """)
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.core.config.settings.database_path", str(path), raising=False)

    repo.init_db()
    old = repo.get_generation("old")
    assert old.markdown_path == "/old.md" and old.pdf_path is None
    assert old.evaluation.quality_score == 77.5 and old.evaluation.dimensions == []
