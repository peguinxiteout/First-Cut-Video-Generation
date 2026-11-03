"""
Runs the ENTIRE blog generation workflow end-to-end with every LLM call
mocked out — mirrors tests/integration/test_workflow_smoke.py's approach
for the brochure workflow. No API keys or network access needed.

The PDF is rendered for real (Playwright Chromium): it is the primary
deliverable, so a smoke test that skipped it would not be smoking much.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.exceptions import ModelProviderError
from src.core.types import QualityGateResult
from src.data import blog_repository as repository
from src.workflows import blog as workflow
from tests.agents.blog import fakes

ALL_DIMENSIONS = ["content_quality", "relevance", "structure_readability", "engagement", "seo"]


class CleanModel(fakes.ScriptedModel):
    """A model that writes a good post first time."""

    def __call__(self, role, fn_name, *args, **kwargs):
        role_val = role.value if hasattr(role, "value") else role
        if role_val == "blog_writer":
            self.calls[role_val] = self.calls.get(role_val, 0) + 1
            return fakes.writer_payload(clean=True)
        return super().__call__(role, fn_name, *args, **kwargs)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    # Never trace test runs into the developer's MLflow store.
    monkeypatch.setattr("src.core.config.settings.mlflow_enabled", False, raising=False)
    monkeypatch.setattr("src.observability.tracing._mlflow", None, raising=False)
    monkeypatch.setattr(workflow, "GENERATED_DIR", tmp_path / "generated_blogs")
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "blog_test.db"), raising=False)
    # The offline placeholder is never printed; stand in for a configured
    # image provider so the hero's rendering is exercised.
    from src.models.image_gen import ImageService

    monkeypatch.setattr(ImageService, "is_mock", property(lambda self: False))
    return tmp_path


@pytest.fixture
def blog_dir(tmp_path):
    return fakes.write_brief(tmp_path / "blogs" / "BLOG-SHELA")


def test_full_blog_workflow_end_to_end(blog_dir, isolated, monkeypatch):
    fake = CleanModel(critic_scores={d: 88 for d in ALL_DIMENSIONS})
    fakes.patch_models(monkeypatch, fake)

    record = workflow.run_blog_workflow(blog_dir)

    assert record.status == "completed", record.error
    assert record.evaluation is not None
    assert record.evaluation.gate_result == QualityGateResult.PASS, record.evaluation.feedback
    assert record.evaluation.quality_score > 0
    assert record.evaluation.critic_mode == "llm+deterministic"
    # The per-dimension breakdown the UI shows, every dimension present.
    assert [d.name for d in record.evaluation.dimensions] == ALL_DIMENSIONS
    assert all(d.llm_score == 88 for d in record.evaluation.dimensions)

    # Exactly one call per agent — no accidental double-calls, no revision.
    # The fact checker is absent: every figure in this post is in the
    # brief, so code settles every claim without a model call.
    assert fake.calls == {"blog_research": 1, "blog_planner": 1, "visual_director": 1,
                          "blog_writer": 1, "blog_seo": 1, "blog_critic": 1}
    assert [r.decision for r in record.revision_history] == ["done"]

    # Workflow Evaluation: every step succeeded, in order.
    assert record.workflow_evaluation is not None
    assert record.workflow_evaluation.all_succeeded
    assert [s.node for s in record.workflow_evaluation.steps] == [
        "research", "planner", "visual", "writer", "fact_check", "editor", "seo", "critic", "render",
    ]

    # The PDF is the deliverable.
    assert record.pdf_path and Path(record.pdf_path).read_bytes()[:4] == b"%PDF"
    assert record.pdf_pages >= 2

    # The Markdown keeps the writer's content.
    md = Path(record.markdown_path).read_text(encoding="utf-8")
    assert "Before you book in Shela: a due-diligence checklist" in md
    assert "- [ ] Confirm the carpet area in writing." in md

    # The preview renders the same article: hero labelled, CTA block.
    html = Path(record.html_path).read_text(encoding="utf-8")
    assert 'class="hero' in html and "Illustration (AI-generated)" in html
    assert "Evening in the west of the city" in html
    assert 'class="cta"' in html and "Book a weekday site visit" in html
    # Placeholder contacts in the brief never reach the page.
    assert "XXXXX" not in html and "https://..." not in html

    # SEO metadata was validated and rebuilt where the model's was unusable.
    assert record.slug == "homes-in-shela-ahmedabad"
    assert 110 <= len(record.meta_description) <= 160

    # Human Review defaults to pending.
    assert record.human_review.status == "pending"

    # Round-trips through the repository, new fields included.
    reloaded = repository.get_generation(record.generation_id)
    assert reloaded is not None
    assert reloaded.evaluation.quality_score == record.evaluation.quality_score
    assert [d.name for d in reloaded.evaluation.dimensions] == ALL_DIMENSIONS
    assert reloaded.pdf_path == record.pdf_path
    assert reloaded.archetype == "locality_guide"


def test_blog_workflow_records_failed_step_and_stops_there(blog_dir, isolated, monkeypatch):
    """A node that crashes (not a model outage — those fall back) fails
    the run at that step, with the steps before it recorded."""
    def crash(*_a, **_k):
        raise RuntimeError("simulated planner crash")

    monkeypatch.setattr("src.agents.blog.planner.run", crash)
    fakes.patch_models(monkeypatch, fakes.ScriptedModel())

    record = workflow.run_blog_workflow(blog_dir)

    assert record.status == "failed"
    assert "simulated planner crash" in record.error
    assert record.workflow_evaluation.failed_steps == ["planner"]
    assert [s.node for s in record.workflow_evaluation.steps] == ["research", "planner"]
    assert record.evaluation is None
    assert record.markdown_path is None and record.pdf_path is None


def test_a_model_outage_does_not_fail_the_run(blog_dir, isolated, monkeypatch):
    """No key / quota exhausted: every agent falls back, the post is still
    written, scored and printed — and the model is asked ONCE, not once
    per agent with back-off sleeps in between."""
    calls = []

    def outage(role, fn_name, *args, **kwargs):
        calls.append(role)
        raise ModelProviderError("GEMINI_API_KEY is not set")

    fakes.patch_models(monkeypatch, outage)
    record = workflow.run_blog_workflow(blog_dir)

    assert record.status == "completed", record.error
    assert record.planner_mode == "fallback" and record.writer_mode == "fallback"
    assert record.evaluation.critic_mode == "deterministic"
    assert len(calls) == 1
    assert record.pdf_path and Path(record.pdf_path).exists()
    # A deterministic draft cannot act on feedback, so no revision loop.
    assert len(record.revision_history) == 1
