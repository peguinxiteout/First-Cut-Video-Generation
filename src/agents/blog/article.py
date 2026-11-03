"""The finished post as one markdown string, assembled in ONE place.

The critic, the SEO agent, the metrics and the editor all need "the whole
article as text"; each used to assemble its own, slightly differently
(the SEO agent's version had no CTA, the critic's had no FAQ), so they
were not evaluating the same document."""
from __future__ import annotations

import re

from src.data.schemas import BlogContent


def full_markdown(content: BlogContent, include_cta: bool = True, include_title: bool = False) -> str:
    parts: list[str] = []
    if include_title and content.title:
        parts.append(f"# {content.title}")
    if content.intro_markdown:
        parts.append(content.intro_markdown)
    if getattr(content, "key_takeaways", None):
        parts.append("## Key takeaways\n\n" + "\n".join(f"- {t}" for t in content.key_takeaways))
    for section in content.sections:
        parts.append(f"## {section.heading}\n\n{section.body_markdown}")
    if getattr(content, "faqs", None):
        faq = "\n\n".join(f"### {f.question}\n\n{f.answer_markdown}" for f in content.faqs)
        parts.append("## Frequently asked questions\n\n" + faq)
    if content.conclusion_markdown:
        heading = getattr(content, "conclusion_heading", "") or ""
        parts.append((f"## {heading}\n\n" if heading else "") + content.conclusion_markdown)
    if include_cta and content.cta_markdown:
        parts.append(f"## {content.cta_heading or 'Next step'}\n\n{content.cta_markdown}")
    return "\n\n".join(p for p in parts if p)


def body_markdown(content: BlogContent) -> str:
    """Running text only — what a reader reads, without headings."""
    parts = [content.intro_markdown, *[s.body_markdown for s in content.sections],
             *[f.answer_markdown for f in getattr(content, "faqs", []) or []],
             content.conclusion_markdown]
    return "\n\n".join(p for p in parts if p)


def plain(markdown: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown or "")
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\[ \]\s+)?", "", text, flags=re.M)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)
    text = re.sub(r"^\s*\|?\s*:?-{3,}.*$", "", text, flags=re.M)
    text = text.replace("|", " ").replace("**", "")
    return text
