"""The Streamlit app: one simple form per solution, and a result shown in
four tabs — Final PDF, Evaluation, Workflow, Human feedback —
with nothing technical in them."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.data.schemas import (  # noqa: E402
    AgentStepResult, EvaluationResult, GenerationRecord, WorkflowEvaluation,
)

APP = str(Path(__file__).resolve().parents[2] / "streamlit_app.py")
TABS = ["Final PDF", "Evaluation", "Workflow", "Human feedback"]


def _app():
    return AppTest.from_file(APP, default_timeout=60)


def _shown(at) -> str:
    """Everything the person can read on the page."""
    parts = []
    for kind in ("markdown", "caption", "error", "warning", "success", "info", "subheader", "title"):
        parts += [str(e.value) for e in getattr(at, kind)]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(parts)


def test_the_app_loads_with_the_brochure_form_and_no_spreadsheet():
    at = _app().run()
    assert not at.exception
    labels = " ".join(w.label for w in at.text_input)
    assert "Project name" in labels and "City" in labels
    shown = _shown(at).lower()
    assert "workbook" not in shown and "xlsx" not in shown and "excel" not in shown
    assert not at.tabs                       # no output area before a generation


def test_missing_essentials_are_named_plainly():
    at = _app().run()
    at.text_input(key="f_name").input("Aurelia Heights")
    at.button[0].click().run()
    warning = " ".join(w.value for w in at.warning)
    assert "the city" in warning and "the category" in warning and "photograph" in warning
    assert "Project name" not in warning


def test_the_blog_form():
    at = _app().run()
    at.radio(key="solution").set_value("Blog post").run()
    assert not at.exception
    assert any("Topic" in w.label for w in at.text_input)
    at.button[0].click().run()
    assert "Please add a topic." in [w.value for w in at.warning]


def test_a_result_shows_exactly_the_five_tabs(tmp_path):
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(b"%PDF-1.4" + b"0" * 6000)
    record = GenerationRecord(
        generation_id="gen_x", property_id="p", status="completed", pdf_path=str(pdf),
        template="Mid-Tier - Portrait", page_count=10,
        evaluation=EvaluationResult(quality_score=86, content_quality=84, visual_design_quality=88,
                                    factual_consistency=100, layout_integrity=100, image_integrity=100,
                                    whitespace_mean=0.14, checks={"whitespace": "pass",
                                                                  "boundaries": "1 problem(s): Page 3: x"}),
        workflow_evaluation=WorkflowEvaluation(steps=[AgentStepResult(node="vision", latency_ms=1200)]),
        agent_reports={"art_director": ["Mid-Tier - Portrait: 10 pages planned"]})
    at = _app()
    at.session_state["brochure_record"] = record
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == TABS
    text = _shown(at)
    assert "Page 3: x" not in text              # a check's internal detail is not shown
    assert "needs attention" in text and "Mid-Tier - Portrait" in text


def test_a_failed_generation_is_explained_plainly():
    record = GenerationRecord(generation_id="gen_y", property_id="p", status="failed",
                              error="RenderingError: Playwright rendering failed: Traceback ...")
    at = _app()
    at.session_state["brochure_record"] = record
    at.run()
    errors = " ".join(e.value for e in at.error)
    assert "couldn't finish this brochure" in errors
    assert "Traceback" not in errors and "Playwright" not in errors


def test_the_format_choice_and_no_brand_colour_field():
    at = _app().run()
    assert at.radio(key="f_orientation").options == ["Landscape", "Portrait"]
    labels = " ".join(w.label for w in at.text_input)
    assert "Brand colour" not in labels and "16:9" not in labels


def test_the_blog_evaluation_shows_five_weighted_parameters_with_a_reason_each(tmp_path):
    from src.data.schemas import BlogDimensionScore, BlogEvaluationResult, BlogGenerationRecord
    from src.evaluation import blog_rubric

    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4" + b"0" * 6000)
    dims = [BlogDimensionScore(name=d["name"], label=d["label"], weight=d["weight"], score=s, threshold=70,
                               passed=s >= 70, summary=f"Reason for {d['label']}.")
            for d, s in zip(blog_rubric.dimension_spec(), (90, 80, 70, 60, 50))]
    evaluation = BlogEvaluationResult(dimensions=dims)
    evaluation.quality_score = blog_rubric.overall(dims)
    record = BlogGenerationRecord(generation_id="blog_x", blog_id="b", status="completed",
                                  pdf_path=str(pdf), evaluation=evaluation)
    at = _app()
    at.session_state["solution"] = "Blog post"
    at.session_state["blog_record"] = record
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == TABS
    text = _shown(at)
    for d in dims:
        assert d.label in text and f"Reason for {d.label}." in text and f"{d.weight:.0%}" in text
    assert evaluation.quality_score == 73.5                      # 27 + 16 + 14 + 9 + 7.5, weighted
    assert "Overall = 30% × 90" in text
