"""The PDF audit reads the rendered pages back and finds a heading left at
the foot of a page — and the renderer can start that section on a new page."""
from __future__ import annotations

from pathlib import Path

from src.data.schemas import BlogContent, BlogSection
from src.rendering.blog_renderer import audit_pdf, render_blog_files, split_lead, _md
from src.rendering.renderer import render_article_pdf


def test_a_stranded_heading_is_found_by_position(tmp_path):
    html = tmp_path / "s.html"
    html.write_text(
        "<html><body style='margin:0'><div style='height:276mm'></div>"
        "<h2 style='margin:0;font:16pt sans-serif'>Stranded heading here</h2>"
        "<p style='font:11pt serif;orphans:3'>" + "Body text. " * 200 + "</p></body></html>", encoding="utf-8")
    pdf = tmp_path / "s.pdf"
    render_article_pdf(html, pdf)
    notes = audit_pdf(pdf, ["Stranded heading here", "Another heading"])["notes"]
    assert any("Stranded heading here" in n and "page 1" in n for n in notes), notes


def test_sections_can_be_forced_onto_a_new_page(tmp_path):
    content = BlogContent(title="T", sections=[BlogSection(heading="Move me", body_markdown="Text.")])
    render_blog_files(content, {}, tmp_path, break_before=frozenset({"Move me"}))
    assert 'class="chapter chapter--prose break-before"' in (tmp_path / "print.html").read_text(encoding="utf-8")


def test_a_short_lead_in_travels_with_the_block_it_introduces():
    lead, rest = split_lead(_md("Here is what is on record:\n\n| A | B |\n|---|---|\n| x | y |\n\nAfter."))
    assert "table-wrap" in lead and "<p>After.</p>" in rest
    lead, rest = split_lead(_md("Intro:\n\n" + "\n".join(f"- item {i}" for i in range(8))))
    assert "<ul>" not in lead                     # a long list is allowed to start on the next page
