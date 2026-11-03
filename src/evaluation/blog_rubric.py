"""The blog evaluation rubric: five parameters, each with a score, a
one-line reason, the evidence behind it and what to change.

    content_quality        depth, accuracy, verified specifics, practical help
    relevance              the brief answered: sections, facts, place, audience
    structure_readability  headings, flow, intro/conclusion, paragraph and
                           sentence length, clean formatting
    engagement             the hook, reader focus, honest tone, a natural CTA
    seo                    keyword placement and density, meta, slug, tags

Names, labels, weights, thresholds and descriptions live in
configs/blog/evaluation.yaml — the critic prompt, this scoring and the UI
all read them from there. Each parameter starts at 100 and loses points for
measured, NAMED problems (blog_metrics.py); each measured problem counts
against ONE parameter only. Every deduction carries a finding and a
suggestion, so a score below threshold turns directly into revision
feedback. When the LLM critic answered, its score for the same parameter is
blended in (`llm_blend`) and its one-line reason becomes the parameter's
explanation; measurements keep the majority so fluent prose cannot talk its
way past a wall of text.

The overall blog score is the weighted sum of the five (`overall`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.agents.blog import blog_config
from src.data.schemas import BlogDimensionScore
from src.evaluation.blog_metrics import Measurement


@dataclass
class _Card:
    name: str
    score: float = 100.0
    findings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def deduct(self, points: float, finding: str, suggestion: str = "") -> None:
        if points <= 0:
            return
        self.score -= points
        self.findings.append(finding)
        if suggestion and suggestion not in self.suggestions:
            self.suggestions.append(suggestion)


def _examples(items: list[str], n: int = 2) -> str:
    items = [i for i in items if i][:n]
    return (" (e.g. " + "; ".join(f"“{i[:70]}”" for i in items) + ")") if items else ""


# What a parameter's one-line reason says when nothing was measured against it.
CLEAN = {
    "content_quality": "Specific, verified content with practical guidance and no untraceable figures.",
    "relevance": "Covers the brief: requested sections, supplied facts, the location and the audience.",
    "structure_readability": "Clear headings and flow, with short paragraphs and sentences.",
    "engagement": "Opens on the reader's question, speaks to them directly and ends with a clear next step.",
    "seo": "Keyword well placed without stuffing; meta title, description and slug in range.",
}


def one_line(name: str, findings: list[str], llm_reason: str = "") -> str:
    """The parameter's explanation in one line: the critic's reason when it
    gave one, else the main measured problem, else what was done well."""
    if llm_reason:
        text = " ".join(str(llm_reason).split())
    elif findings:
        text = findings[0].rstrip(".") + (f" (+{len(findings) - 1} more)." if len(findings) > 1 else ".")
    else:
        text = CLEAN.get(name, "No problems measured.")
    return text if len(text) <= 180 else text[:177].rsplit(" ", 1)[0] + "…"


def score_dimensions(m: Measurement, llm_scores: Optional[dict] = None,
                     has_research_places: bool = False, location_known: bool = False,
                     brand_given: bool = False, llm_reasons: Optional[dict] = None) -> list[BlogDimensionScore]:
    cfg = blog_config.rubric()
    limits = cfg.get("limits", {})
    blend = float(cfg.get("llm_blend", 0.4))
    max_para = limits.get("paragraph_max_words", 70)
    cards = {name: _Card(name) for name in cfg["dimensions"]}

    # --- content quality: depth, accuracy, usefulness -------------------------
    c = cards["content_quality"]
    if m["untraceable_figures"]:
        c.deduct(min(60, 25 * m["untraceable_figures"]),
                 "Figures not traceable to the brief or research: " + ", ".join(m.detail("untraceable_figures")),
                 "Use only figures from the brief or research; otherwise state the point qualitatively.")
    if m["inconsistent_figures"]:
        c.deduct(15 * m["inconsistent_figures"], "Inconsistent figures: " + "; ".join(m.detail("inconsistent_figures")),
                 "State each figure the same way everywhere.")
    if m["implausible_units"]:
        c.deduct(15 * m["implausible_units"], "Implausible values/units: " + ", ".join(m.detail("implausible_units")),
                 "Check units (sq ft vs sq m, lakh vs crore, km vs min) and magnitudes.")
    if m["placeholders"]:
        c.deduct(15 * m["placeholders"], "Placeholders found: " + ", ".join(m.detail("placeholders")),
                 "Remove placeholders such as example.com, [link], XXXX.")
    if m["prohibited_claims"]:
        c.deduct(25 * m["prohibited_claims"], "Prohibited claims: " + ", ".join(m.detail("prohibited_claims")),
                 "Never promise returns or appreciation.")
    if not m["has_specifics"]:
        c.deduct(15, "No verified specifics: neither the brief nor research supplied facts, so the post can "
                     "only give general guidance.",
                 "Add facts the publisher can vouch for (prices, distances, approvals) or enable research.")
    if m["word_count"] < 500:
        c.deduct(15, f"Thin article ({m['word_count']:.0f} words).",
                 "Develop each section with specific reasoning, examples and decision guidance.")
    if m["checklist_items"] == 0 and m["ordered_lists"] == 0:
        c.deduct(10, "No checklist or step sequence to act on.",
                 "Add a checklist or numbered steps the reader can use directly.")
    if m["tables"] == 0:
        c.deduct(6, "No table: comparisons or key facts are not scannable.",
                 "Add one table that compares options or lists the key facts with their sources.")
    if m["takeaways"] == 0 and m["faqs"] == 0:
        c.deduct(4, "No key takeaways or FAQ to anchor the answer.",
                 "Open with 3-5 key takeaways that answer the search query directly.")
    if m["unverified_admissions"] > 1:
        c.deduct(min(12, 3 * (m["unverified_admissions"] - 1)),
                 f"{m['unverified_admissions']:.0f} places where the post says it could not verify something.",
                 "Supply the missing local facts so those sections can be specific.")
    if not m["verification_language"]:
        c.deduct(6, "The reader is never told what to verify or where.",
                 "Tell the reader what to verify and where (e.g. the state RERA portal).")
    if m["figures"] and m["references"] == 0:
        c.deduct(8, "Figures are stated with no sources section.", "List sources for the facts used.")
    if m["stock_phrases"]:
        c.deduct(4 * m["stock_phrases"], "Stock phrasing: " + ", ".join(m.detail("stock_phrases")),
                 "Replace stock phrases with concrete statements.")
    if m["repeated_ngram_share"] > 0.08:
        c.deduct(10, f"Repetitive wording ({m['repeated_ngram_share']:.0%} of phrases repeat).",
                 "Cut repeated points; each section should add something new.")

    # --- relevance: the brief answered ------------------------------------------
    c = cards["relevance"]
    if m["missing_sections"]:
        c.deduct(min(45, 15 * m["missing_sections"]),
                 "Requested sections not covered: " + ", ".join(m.detail("missing_sections")),
                 "Cover every requested section explicitly, with its own heading.")
    if m["brief_facts"] and m["brief_fact_coverage"] < 1:
        missing = m["brief_facts"] * (1 - m["brief_fact_coverage"])
        c.deduct(min(30, 7 * missing), f"{missing:.0f} of the brief's {m['brief_facts']:.0f} facts are not used.",
                 "Use every fact supplied in the brief where it helps the reader decide.")
    if location_known and m["location_mentions"] < 3:
        c.deduct(12, "The place is barely mentioned; the post reads as interchangeable with any city.",
                 "Anchor each section in the specific location — its roads, landmarks and facts.")
    if m["stray_places"]:
        c.deduct(min(24, 8 * m["stray_places"]), "Irrelevant places mentioned: " + ", ".join(m.detail("stray_places")),
                 "Remove references to places the brief is not about.")
    if m["audience_mentions"] == 0:
        c.deduct(15, "The stated audience is never addressed.",
                 "Speak to the target audience's specific situation and priorities.")
    if has_research_places and m["research_entities_used"] == 0:
        c.deduct(5, "None of the researched nearby places are named.",
                 "Name the real nearby places from the research where they matter to the reader.")
    if m["competitor_mentions"]:
        c.deduct(10 * m["competitor_mentions"], "Competitors are cited or promoted: "
                 + ", ".join(m.detail("competitor_mentions")[:3]),
                 "Do not cite or link listing portals or competitors; prefer official sources.")

    # --- structure & readability ---------------------------------------------------
    c = cards["structure_readability"]
    min_s, max_s = limits.get("min_sections", 4), limits.get("max_sections", 10)
    if m["long_paragraphs"]:
        c.deduct(min(25, 50 * m["long_paragraph_share"] + 4),
                 f"{m['long_paragraphs']:.0f} paragraph(s) exceed {max_para} words or "
                 f"{limits.get('paragraph_max_sentences', 4)} sentences{_examples(m.detail('long_paragraphs'))}.",
                 "Keep paragraphs to 2-3 sentences and under ~60 words.")
    if m["avg_sentence_words"] > 22:
        c.deduct(min(15, 3 * (m["avg_sentence_words"] - 22)),
                 f"Average sentence is {m['avg_sentence_words']:.0f} words.",
                 "Aim for 15-20 words per sentence.")
    if m["long_sentence_share"] > 0.12:
        c.deduct(min(12, 100 * (m["long_sentence_share"] - 0.12)),
                 f"{m['long_sentence_share']:.0%} of sentences run past "
                 f"{limits.get('sentence_long_words', 30)} words{_examples(m.detail('long_sentences'))}.",
                 "Break the longest sentences in two.")
    if m["duplicate_sentences"]:
        c.deduct(min(18, 6 * m["duplicate_sentences"]),
                 f"{m['duplicate_sentences']:.0f} repeated sentence(s){_examples(m.detail('duplicate_sentences'))}.",
                 "Say each thing once, in the section where it belongs.")
    if m["repeated_opener_share"] > 0.12 and m["sentences"] > 12:
        c.deduct(6, "Many sentences open the same way: " + ", ".join(m.detail("repeated_opener")),
                 "Vary sentence openings.")
    if m["section_count"] < min_s:
        c.deduct(15, f"Only {m['section_count']:.0f} sections.", f"Plan {min_s}-{max_s} sections with distinct jobs.")
    if m["section_count"] > max_s:
        c.deduct(10, f"{m['section_count']:.0f} sections fragment the argument.", "Merge overlapping sections.")
    if m["generic_headings"]:
        c.deduct(6 * m["generic_headings"], "Generic headings: " + ", ".join(m.detail("generic_headings")),
                 "Make every heading say something specific.")
    if m["long_headings"]:
        c.deduct(4 * m["long_headings"], "Over-long headings.", "Keep headings under 12 words.")
    if m["duplicate_headings"]:
        c.deduct(10, "Duplicate headings.", "Each section needs a distinct job and heading.")
    if m["empty_sections"]:
        c.deduct(10 * m["empty_sections"], f"{m['empty_sections']:.0f} section(s) are near-empty.",
                 "Develop or remove thin sections.")
    if not m["has_intro"]:
        c.deduct(15, "Missing or very short introduction.", "Open with an introduction that frames the decision.")
    if not m["has_conclusion"]:
        c.deduct(15, "Missing or very short conclusion.", "Close with a decisive conclusion.")
    elif not m["conclusion_decisive"]:
        c.deduct(10, "The conclusion summarises instead of concluding.",
                 "Close with a clear verdict: who this suits, who should look elsewhere, and the first check.")
    if m["lists"] + m["tables"] == 0:
        c.deduct(12, "No lists or tables: the article is all running prose.",
                 "Turn enumerations into bullet lists and comparisons into a table.")
    if m["markup_leaks"]:
        c.deduct(10 * m["markup_leaks"], "Raw markdown would show in the output: " + ", ".join(m.detail("markup_leaks")),
                 "Use clean markdown: one list item per line, balanced bold markers, no stray # or |---|.")
    if m["bold_share"] > limits.get("bold_word_share_max", 0.03):
        c.deduct(6, f"Too much bold text ({m['bold_share']:.0%} of words).",
                 "Bold at most one short phrase per section.")

    # --- audience engagement --------------------------------------------------------
    c = cards["engagement"]
    if m["hook_generic"]:
        c.deduct(20, "The opening is generic (a definition or cliché rather than the reader's problem).",
                 "Open with the reader's real tension or trade-off in the first sentence.")
    if m["you_per_100_words"] < 0.8:
        c.deduct(12, "The post rarely addresses the reader directly.",
                 "Write to the reader ('you', 'your budget', 'your commute') with relatable situations.")
    if m["promotional_terms"]:
        c.deduct(min(30, 6 * m["promotional_terms"]), "Sales language: " + ", ".join(m.detail("promotional_terms")),
                 "Persuade with evidence and honest trade-offs instead of sales adjectives.")
    if not m["cta_present"]:
        c.deduct(25, "No call to action.", "End with one natural, specific next step for the reader.")
    if m["cta_placeholder"]:
        c.deduct(25, "The call to action contains placeholder contact details.",
                 "Never write placeholder links or contacts; write the CTA without contact details.")
    if brand_given and m["brand_mentions"] > blog_config.limits().get("brand_mentions_max", 4):
        c.deduct(10, f"The brand is mentioned {m['brand_mentions']:.0f} times.",
                 "Mention the brand once or twice, where it is relevant (the CTA), not throughout.")
    if m["faqs"] == 0 and m["question_headings"] == 0 and m["callouts"] == 0:
        c.deduct(5, "Little variety in presentation.",
                 "Use a tip/callout box, a question-led heading or a short FAQ where they help.")

    # --- SEO optimisation --------------------------------------------------------------
    c = cards["seo"]
    if not m["keyword_in_title"]:
        c.deduct(15, "Primary keyword missing from the title.", "Put the primary keyword in the title naturally.")
    if not m["keyword_in_intro"]:
        c.deduct(10, "Primary keyword missing from the first 100 words.",
                 "Use the primary keyword once in the opening paragraph.")
    if not m["keyword_in_heading"]:
        c.deduct(8, "Primary keyword in no section heading.", "Use the keyword (or a close variant) in one heading.")
    if not m["keyword_in_meta"]:
        c.deduct(8, "Primary keyword missing from the meta title/description.",
                 "Include the primary keyword in the meta description.")
    density_max = limits.get("keyword_density_max", 2.5)
    if m["keyword_density"] > density_max:
        c.deduct(min(25, 8 * (m["keyword_density"] - density_max) + 5),
                 f"Keyword density {m['keyword_density']:.1f}% (stuffing).",
                 f"Keep the primary keyword under {density_max}% of words; use natural variants.")
    lo, hi = limits.get("meta_title_min", 30), limits.get("meta_title_max", 60)
    if not lo <= m["meta_title_length"] <= hi:
        c.deduct(8, f"Meta title is {m['meta_title_length']:.0f} characters.", f"Meta title: {lo}-{hi} characters.")
    lo, hi = limits.get("meta_description_min", 110), limits.get("meta_description_max", 160)
    if not lo <= m["meta_description_length"] <= hi:
        c.deduct(8, f"Meta description is {m['meta_description_length']:.0f} characters.",
                 f"Meta description: {lo}-{hi} characters, a specific promise of what the reader learns.")
    if not m["slug_ok"]:
        c.deduct(6, "Slug missing or malformed.", "Short, lowercase, hyphenated slug built from the keyword.")
    if m["tags"] < 3:
        c.deduct(5, "Fewer than 3 topic tags.", "Add 3-6 topic tags (location, topic, audience).")
    if m["secondary_coverage"] < 0.5:
        c.deduct(6, "Most secondary keywords are not covered.", "Work secondary keywords into headings or body naturally.")

    out = []
    reasons = llm_reasons or {}
    for name, card in cards.items():
        dim_cfg = cfg["dimensions"][name]
        deterministic = max(0.0, min(100.0, card.score))
        llm = None
        if llm_scores and isinstance(llm_scores.get(name), (int, float)):
            llm = max(0.0, min(100.0, float(llm_scores[name])))
        final = deterministic if llm is None else (1 - blend) * deterministic + blend * llm
        reason = str(reasons.get(name) or "") if llm is not None else ""
        out.append(BlogDimensionScore(
            name=name, label=dim_cfg.get("label", name), score=round(final, 1),
            deterministic_score=round(deterministic, 1), llm_score=llm,
            weight=float(dim_cfg.get("weight", 0)), threshold=float(dim_cfg.get("threshold", 0)),
            passed=final >= float(dim_cfg.get("threshold", 0)),
            summary=one_line(name, card.findings, reason),
            findings=card.findings, suggestions=card.suggestions))
    return out


def overall(dimensions: list[BlogDimensionScore]) -> float:
    """The weighted blog score: the sum of weight x score (weights sum to 1)."""
    total_weight = sum(d.weight for d in dimensions) or 1.0
    return round(sum(d.score * d.weight for d in dimensions) / total_weight, 2)


def dimension_spec() -> list[dict]:
    """name, label, weight and description of every parameter, in order —
    for the critic prompt and the UI, from the one config."""
    return [{"name": n, "label": d.get("label", n), "weight": float(d.get("weight", 0)),
             "description": d.get("description", "")} for n, d in blog_config.rubric()["dimensions"].items()]


def feedback(dimensions: list[BlogDimensionScore], llm_notes: Optional[dict] = None) -> list[str]:
    """Parameter-level revision instructions, worst first."""
    lines = []
    for d in sorted(dimensions, key=lambda d: d.score - d.threshold):
        if d.passed and d.score >= d.threshold + 5:
            continue
        notes = list(d.suggestions)
        if llm_notes and llm_notes.get(d.name):
            notes.append(str(llm_notes[d.name]))
        if not notes:
            continue
        lines.append(f"{d.label} ({d.score:.0f}/100, target {d.threshold:.0f}): " + " ".join(notes))
    return lines
