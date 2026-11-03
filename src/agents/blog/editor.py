"""Editor Agent — the markdown normaliser and validator, applied to a whole
draft (deterministic; no model).

Runs after the fact check and before SEO, on EVERY draft, whoever wrote
it. Two halves:

FIX what is mechanically fixable
  * formatting: collapsed bullets, missing blank lines, heading levels,
    stray **/##/|---|, malformed tables, bold density, walls of text
    (src/agents/blog/normaliser.py);
  * links and contacts: placeholder links and contact details removed,
    links to anything that is not a verified source unlinked;
  * accuracy: sentences whose figures trace to neither the brief nor the
    research are removed (and the fact checker's model-judged unsupported
    claims with them) — an untraceable number is never printed;
  * compliance and relevance: return promises and sentences about cities
    the brief never mentioned are removed;
  * repetition: a sentence that repeats an earlier one is removed.

REPORT what is not: generic headings, sales language, keyword stuffing,
inconsistent figures, implausible units, a missing CTA. Those need a
writer, so they go into the validation report and from there into the
revision feedback.
"""
from __future__ import annotations

import re
from typing import Optional

from src.agents.blog import article, brief, composer, fact_check as fact_check_mod
from src.agents.blog.normaliser import (
    NormaliseResult, blocks_of, block_kind, find_placeholders, markup_leaks, normalise_markdown,
    remove_sentences, split_sentences,
)
from src.data.schemas import (
    BlogContent, BlogFAQ, BlogInput, BlogValidationIssue, BlogValidationReport, FactCheckResult, ResearchPack,
)
from src.evaluation import blog_metrics


def _clean_inline(text: str, max_words: int = 0) -> str:
    """A title, heading, question or takeaway: one line, no markup."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text or "")
    text = re.sub(r"\*\*|__|`|^#+\s*|^\s*[-*+]\s+|^\s*\d+[.)]\s+", "", text.strip())
    text = re.sub(r"(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)", "", text)
    text = " ".join(text.split()).strip(" :")
    if max_words and len(text.split()) > max_words:
        text = " ".join(text.split()[:max_words]).rstrip(",;:")
    return text


def allowed_urls(blog_input: BlogInput, research: Optional[ResearchPack], content: BlogContent) -> list[str]:
    urls = [brief.contact_details(blog_input)["url"]]
    urls += [r.url for r in content.references if r.url]
    urls += [r.url for r in composer.build_references(blog_input, research) if r.url]
    if research is not None:
        urls += list(research.sources) + [r.url for r in research.official_sources if r.url]
    return [u for u in dict.fromkeys(urls) if u]


def run(blog_input: BlogInput, content: BlogContent, research: Optional[ResearchPack] = None,
        fact_check: Optional[FactCheckResult] = None) -> tuple[BlogContent, BlogValidationReport]:
    report = BlogValidationReport()
    content = content.model_copy(deep=True)
    contacts = [v for v in brief.contact_details(blog_input).values() if v]
    urls = allowed_urls(blog_input, research, content)

    def norm(md: str, allow_headings: bool, where: str) -> str:
        result: NormaliseResult = normalise_markdown(md or "", allowed_urls=urls, allowed_contacts=contacts,
                                                     allow_headings=allow_headings)
        for fix in result.fixes:
            if fix not in report.fixes:
                report.fixes.append(fix)
        for code, severity, message in result.issues:
            report.issues.append(BlogValidationIssue(code=code, severity=severity, message=message, location=where))
        return result.text

    # --- 1. formatting ---------------------------------------------------------
    content.title = _clean_inline(content.title, 16)
    content.intro_markdown = norm(content.intro_markdown, False, "introduction")
    for section in content.sections:
        section.heading = _clean_inline(section.heading, 14)
        section.body_markdown = norm(section.body_markdown, True, section.heading)
    content.conclusion_heading = _clean_inline(content.conclusion_heading, 12)
    content.conclusion_markdown = norm(content.conclusion_markdown, False, "conclusion")
    content.cta_heading = _clean_inline(content.cta_heading, 10)
    content.cta_markdown = norm(content.cta_markdown, False, "call to action")
    content.faqs = [BlogFAQ(question=_clean_inline(f.question, 18), answer_markdown=norm(f.answer_markdown, False, "FAQ"))
                    for f in content.faqs if f.question.strip()]
    content.key_takeaways = [_clean_inline(t, 32) for t in content.key_takeaways if _clean_inline(t)]

    # --- 2. accuracy, compliance, relevance -------------------------------------
    evidence = fact_check_mod.evidence_for(blog_input, research)
    judged_unsupported = set()
    if fact_check is not None and fact_check.unsupported and not any(
            "could not reach" in n for n in fact_check.notes):
        judged_unsupported = {_key(c) for c in fact_check.unsupported}
    allowed_geo = blog_metrics.allowed_geography(blog_input, research)

    def offending(sentence: str) -> str:
        if fact_check_mod.untraceable_figures(sentence, evidence):
            return "untraceable"
        if _key(sentence) in judged_unsupported:
            return "unsupported"
        if blog_metrics.prohibited_claims(sentence):
            return "prohibited"
        if blog_metrics.stray_places(sentence, allowed_geo):
            return "stray_place"
        return ""

    removed: dict[str, list[str]] = {}

    def scrub(md: str) -> str:
        new, gone = remove_sentences(md, lambda s: bool(offending(s)))
        for sentence in gone:
            removed.setdefault(offending(sentence) or "untraceable", []).append(sentence)
        return new

    content.intro_markdown = scrub(content.intro_markdown)
    for section in content.sections:
        section.body_markdown = scrub(section.body_markdown)
    content.conclusion_markdown = scrub(content.conclusion_markdown)
    content.cta_markdown = scrub(content.cta_markdown)
    for faq in content.faqs:
        faq.answer_markdown = scrub(faq.answer_markdown)
    content.faqs = [f for f in content.faqs if f.answer_markdown.strip()]
    kept_takeaways = []
    for t in content.key_takeaways:
        reason = offending(t)
        if reason:
            removed.setdefault(reason, []).append(t)
        else:
            kept_takeaways.append(t)
    content.key_takeaways = kept_takeaways
    for heading_owner in content.sections:
        figures = fact_check_mod.untraceable_figures(heading_owner.heading, evidence)
        if figures:
            report.issues.append(BlogValidationIssue(
                code="heading_figure", severity="warning", location=heading_owner.heading,
                message=f"Heading contains an untraceable figure ({', '.join(figures)})."))
    labels = {"untraceable": "a figure that traces to neither the brief nor the research",
              "unsupported": "a claim the fact check could not support",
              "prohibited": "a promise of returns (not allowed in property marketing)",
              "stray_place": "a place the brief is not about"}
    for reason, sentences in removed.items():
        report.removed_claims.extend(sentences)
        report.fixes.append(f"Removed {len(sentences)} sentence(s) containing {labels[reason]}.")
        for s in sentences[:6]:
            report.issues.append(BlogValidationIssue(code=f"removed_{reason}", severity="info",
                                                     message=f'Removed: "{s[:110]}"'))

    # --- 3. repetition ------------------------------------------------------------
    seen: list[str] = []
    dropped = 0

    def dedupe(md: str) -> str:
        nonlocal dropped

        def is_dupe(sentence: str) -> bool:
            if len(sentence.split()) < 7:
                return False
            if any(blog_metrics.near_duplicate(sentence, s) for s in seen):
                return True
            seen.append(sentence)
            return False

        new, gone = remove_sentences(md, is_dupe, kinds={"text", "list", "quote"})
        dropped += len(gone)
        return new

    content.intro_markdown = dedupe(content.intro_markdown)
    for section in content.sections:
        section.body_markdown = dedupe(section.body_markdown)
    for faq in content.faqs:
        faq.answer_markdown = dedupe(faq.answer_markdown)
    content.faqs = [f for f in content.faqs if f.answer_markdown.strip()]
    content.conclusion_markdown = dedupe(content.conclusion_markdown)
    if dropped:
        report.fixes.append(f"Removed {dropped} repeated sentence(s).")

    # --- 4. structure that the fixes above may have thinned -------------------------
    kept_sections = []
    for section in content.sections:
        words = len(article.plain(section.body_markdown).split())
        if words == 0:
            report.issues.append(BlogValidationIssue(code="empty_section", severity="warning",
                                                     location=section.heading,
                                                     message=f'Section "{section.heading}" was empty after editing '
                                                             "and was removed."))
            continue
        if words < 25:
            report.issues.append(BlogValidationIssue(code="thin_section", severity="warning",
                                                     location=section.heading,
                                                     message=f'Section "{section.heading}" is very short.'))
        kept_sections.append(section)
    content.sections = kept_sections
    seen_headings: set[str] = set()
    for section in content.sections:
        if section.heading.lower() in seen_headings:
            report.issues.append(BlogValidationIssue(code="duplicate_heading", severity="warning",
                                                     location=section.heading,
                                                     message=f'Heading "{section.heading}" is used twice.'))
        seen_headings.add(section.heading.lower())
    if len(article.plain(content.cta_markdown).split()) < 6:
        heading, body = composer.default_cta(blog_input, content.archetype)
        content.cta_heading, content.cta_markdown = heading, body
        report.fixes.append("Replaced a call to action that was empty or only placeholders with one built "
                            "from the brief.")
    if not content.key_takeaways:
        content.key_takeaways = derive_takeaways(content)
        if content.key_takeaways:
            report.fixes.append("Added key takeaways from the opening line of each section.")

    # --- 5. what only a writer can fix -------------------------------------------
    report.issues.extend(validate(blog_input, content, research, evidence))
    return content, report


def derive_takeaways(content: BlogContent, n: int = 4) -> list[str]:
    out = []
    for section in content.sections:
        for block in blocks_of(section.body_markdown):
            if block_kind(block) != "text":
                continue
            first = (split_sentences(article.plain(block)) or [""])[0]
            if 6 <= len(first.split()) <= 30 and not blog_metrics.promotional_terms(first):
                out.append(first)
            break
        if len(out) >= n:
            break
    return out


def validate(blog_input: BlogInput, content: BlogContent, research: Optional[ResearchPack],
             evidence: Optional[dict] = None) -> list[BlogValidationIssue]:
    """Checks the Editor cannot fix by itself. Severity "error" means the
    post must not be published as is."""
    issues: list[BlogValidationIssue] = []

    def add(code: str, severity: str, message: str, location: str = "") -> None:
        issues.append(BlogValidationIssue(code=code, severity=severity, message=message, location=location))

    full = article.full_markdown(content)
    rendered_like = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", article.plain(full))
    rendered_like = re.sub(r"\*\*([^*]+)\*\*", r"\1", rendered_like)
    for leak in markup_leaks(rendered_like):
        add("markup_leak", "error", f"Raw markdown would reach the page: {leak}.")
    for placeholder in find_placeholders(full):
        add("placeholder", "error", f"Placeholder text remains: “{placeholder}”.")
    for heading in blog_metrics.generic_headings([s.heading for s in content.sections]):
        add("generic_heading", "warning", f'Generic heading: "{heading}". Say what the section argues.', heading)
    for section in content.sections:
        if len(section.heading.split()) > 12:
            add("long_heading", "warning", f'Heading is long ({len(section.heading.split())} words).', section.heading)
    promo = blog_metrics.promotional_terms(full)
    if promo:
        add("promotional", "warning", "Sales language: " + ", ".join(promo) + ".")
    keyword = content.primary_keyword or brief.primary_keyword(blog_input)
    if keyword:
        from src.agents.blog import blog_config

        density = blog_metrics.keyword_density(article.plain(full), keyword)
        if density > blog_config.limits().get("keyword_density_max", 2.5):
            add("keyword_stuffing", "warning", f'Primary keyword "{keyword}" is overused ({density:.1f}% of words).')
    for item in blog_metrics.inconsistent_figures(full):
        add("inconsistent_figure", "warning", f"Inconsistent figures for {item}.")
    for item in blog_metrics.implausible_units(full):
        add("implausible_unit", "warning", f"Implausible value or unit: {item}.")
    if not content.cta_markdown.strip():
        add("missing_cta", "warning", "No call to action.")
    if len(content.sections) < 3:
        add("few_sections", "warning", f"Only {len(content.sections)} section(s).")
    return issues


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def as_revision_note(report: BlogValidationReport) -> list[str]:
    """What the writer should do differently, from the validation report."""
    lines = []
    if report.removed_claims:
        lines.append("These sentences were REMOVED because their figures could not be traced to the brief or the "
                     "research. Do not reintroduce them; make the point without the figure, attribute it to a "
                     "source in the evidence, or say estimates vary:")
        lines += [f'- "{c[:140]}"' for c in report.removed_claims[:8]]
    unresolved = [i for i in report.issues if i.severity in ("error", "warning")]
    if unresolved:
        lines.append("Fix these issues the editor could not fix mechanically:")
        lines += [f"- {i.message}" for i in unresolved[:10]]
    return lines
