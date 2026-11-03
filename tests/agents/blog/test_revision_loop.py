"""The evaluation feedback loop, end to end: a bad first draft is scored,
specific dimension-level feedback goes back to the writer, the revision
is scored again, the loop is bounded, and the BEST draft is published."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.types import QualityGateResult
from src.workflows import blog as workflow
from tests.agents.blog import fakes

ALL = ["content_quality", "relevance", "structure_readability", "engagement", "seo"]


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "GENERATED_DIR", tmp_path / "generated_blogs")
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "loop.db"), raising=False)

    def _run(fake):
        fakes.patch_models(monkeypatch, fake)
        return workflow.run_blog_workflow(fakes.write_brief(tmp_path / "blogs" / "B"))
    return _run


def test_a_bad_draft_is_sent_back_with_specific_feedback_and_the_revision_wins(run):
    fake = fakes.ScriptedModel(critic_scores={d: 85 for d in ALL})
    record = run(fake)

    assert record.status == "completed", record.error
    assert fake.calls["blog_writer"] == 2
    first, second = record.revision_history
    assert first.decision == "revise" and first.failing_dimensions
    assert second.quality_score > first.quality_score
    assert second.decision == "done"
    assert record.evaluation.gate_result == QualityGateResult.PASS
    assert record.writer_mode == "llm"

    # The writer was told what to fix, dimension by dimension, including
    # the invented figure the Editor had to remove.
    revision_prompt = fake.prompts["blog_writer"][1]
    assert "REVISION" in revision_prompt
    assert "Structure & readability (" in revision_prompt and "target" in revision_prompt
    assert "40%" in revision_prompt and "REMOVED" in revision_prompt

    # What was published is the revision, cleaned and printed.
    md = Path(record.markdown_path).read_text(encoding="utf-8")
    assert "Before you book in Shela" in md and "40%" not in md and "XXXXX" not in md
    assert record.pdf_path and Path(record.pdf_path).exists()


def test_the_loop_is_bounded_and_keeps_the_best_draft(run):
    class NeverImproves(fakes.ScriptedModel):
        def __call__(self, role, fn, *a, **k):
            role_val = role.value if hasattr(role, "value") else role
            if role_val == "blog_writer":
                self.calls[role_val] = self.calls.get(role_val, 0) + 1
                self.prompts.setdefault(role_val, []).append(a[0])
                return fakes.writer_payload(clean=False)
            return super().__call__(role, fn, *a, **k)

    fake = NeverImproves(critic_scores={d: 40 for d in ALL})
    record = run(fake)

    from src.evaluation.blog_scoring import max_revisions
    assert fake.calls["blog_writer"] == 1 + max_revisions()
    assert record.revision_history[-1].decision.startswith("done: revision limit")
    assert record.evaluation.gate_result == QualityGateResult.REVISE
    # Even an unimproved draft goes out clean: the Editor fixed what it could.
    md = Path(record.markdown_path).read_text(encoding="utf-8")
    assert "XXXXX" not in md and "example.com" not in md and "40%" not in md
    assert "- Close to SP Ring Road\n- Several CBSE schools" in md
    assert record.validation and record.validation.removed_claims


def test_a_writer_outage_mid_run_falls_back_and_stops_revising(run):
    fake = fakes.ScriptedModel(critic_scores={d: 30 for d in ALL}, writer_fails=True)
    record = run(fake)
    assert record.status == "completed", record.error
    assert record.writer_mode == "fallback"
    assert len(record.revision_history) == 1
    assert record.revision_history[0].decision in ("done",) or \
        record.revision_history[0].decision.startswith("skipped")
    assert Path(record.pdf_path).exists()
