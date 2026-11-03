"""Offline end to end: no API key, no network. Two briefs of different
archetypes go through the whole workflow on the deterministic path and
come out as PDFs — the primary deliverable — that read cleanly."""
from __future__ import annotations

import json
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from src.core.exceptions import ModelProviderError
from src.workflows import blog as workflow
from tests.agents.blog import fakes

LOCALITY = fakes.SHELA_BRIEF

COMPARISON = {
    "blog_id": "BLOG-WF",
    "topic": "Whitefield vs Sarjapur Road, Bengaluru: which suits IT professionals?",
    "audience": "first-time buyers working in IT",
    "sections": ["comparison", "pros_cons", "pricing_value", "buyer_considerations"],
    "local_facts": ["Whitefield has an operational Purple Line metro connection",
                    "Sarjapur Road has no operational metro line yet",
                    "Whitefield 2BHK resale homes are listed at ₹85 lakh–1.1 crore",
                    "Sarjapur Road 2BHK new launches are listed at ₹75–95 lakh"],
    "cta_email": "advice@northstar-realty.in",
    "brand_name": "Northstar Realty",
}


def no_model(role, fn_name, *a, **k):
    raise ModelProviderError("GEMINI_API_KEY is not set")


def pdf_text(path: str) -> str:
    pdf = pdfium.PdfDocument(path)
    try:
        return "\n".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
    finally:
        pdf.close()


@pytest.mark.parametrize("brief, archetype, must_contain", [
    (LOCALITY, "locality_guide", ["Key takeaways", "Before you book in Shela", "GujRERA",
                                  "Sources and where to verify"]),
    (COMPARISON, "comparison", ["Whitefield vs Sarjapur Road at a glance", "Has an operational Purple Line",
                                "Which one fits you", "advice@northstar-realty.in"]),
])
def test_offline_brief_becomes_a_clean_pdf(tmp_path, monkeypatch, brief, archetype, must_contain):
    monkeypatch.setattr(workflow, "GENERATED_DIR", tmp_path / "generated_blogs")
    monkeypatch.setattr("src.core.config.settings.database_path", str(tmp_path / "pdf.db"), raising=False)
    fakes.patch_models(monkeypatch, no_model)
    blog_dir = tmp_path / "blogs" / brief["blog_id"]
    blog_dir.mkdir(parents=True)
    (blog_dir / "blog_input.json").write_text(json.dumps(brief), encoding="utf-8")

    record = workflow.run_blog_workflow(blog_dir)

    assert record.status == "completed", record.error
    assert record.archetype == archetype
    assert record.pdf_path and Path(record.pdf_path).read_bytes()[:4] == b"%PDF"
    assert record.pdf_pages >= 3
    text = pdf_text(record.pdf_path)
    squashed = "".join(text.split()).lower()        # letter-spaced labels extract with gaps
    for snippet in must_contain:
        assert "".join(snippet.split()).lower() in squashed, snippet
    # Nothing raw, nothing fake, nothing stranded.
    for bad in ("**", "|---", "](", "example.com", "XXXX", "https://..."):
        assert bad not in text, bad
    assert not [n for n in record.render_report if "bottom of page" in n or "Raw markup" in n], record.render_report
    # Scored on every dimension, deterministically.
    assert len(record.evaluation.dimensions) == 5
    assert record.evaluation.critic_mode == "deterministic"
    assert all(step.status == "success" for step in record.workflow_evaluation.steps)
