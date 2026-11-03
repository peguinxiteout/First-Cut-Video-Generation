"""Phase 2: the deterministic editorial checks that encode the
original-vs-generated blog evaluation's findings."""
from __future__ import annotations

from src.agents.blog import brief
from src.data.schemas import BlogContent, BlogInput, BlogSection
from src.evaluation.blog_deterministic import (
    banned_phrases_used, compute_editorial_integrity, stuffed_keywords, uncovered_sections,
    unsupported_figures,
)


def _input(**kw):
    base = dict(blog_id="B", topic="Buying in Noida Sector 128")
    base.update(kw)
    return BlogInput(**base)


def test_invented_figures_are_flagged_but_provided_ones_are_not():
    """The evaluation wanted concrete data; without a data source that
    just means invented data. Figures must trace to the author's facts."""
    bi = _input(local_facts=["Metro station is 2 km away", "Prices around ₹12,500 per sq ft"])
    post = ("The metro is 2 km away and prices sit near ₹12,500 per sq ft. "
            "Land values have risen 14% annually and the expressway saves 25 minutes.")
    flagged = unsupported_figures(post, bi)
    assert any("14%" in f for f in flagged)
    assert any("25 minutes" in f for f in flagged)
    assert not any("12,500" in f or "2 km" in f for f in flagged)


def test_small_bare_counts_are_not_treated_as_factual_claims():
    assert unsupported_figures("Three towers and 4 bedrooms.", _input()) == []


def test_keyword_stuffing_is_detected():
    """The original blog's 'flats for sale Electronic City Phase 1' pattern."""
    phrase = "flats for sale electronic city"
    post = " ".join([f"Looking at {phrase}?"] * 5)
    assert stuffed_keywords(post, [phrase, "home loans"]) == [phrase]
    assert stuffed_keywords(f"One mention of {phrase}.", [phrase]) == []


def test_stock_ai_phrasing_is_detected_with_word_boundaries():
    assert banned_phrases_used("Nestled in greenery, it boasts a pool.") == ["nestled", "boasts"]
    assert banned_phrases_used("The unlocked gate.") == []  # 'unlock' is not in 'unlocked' as a word


def test_section_coverage_uses_planner_assignment_or_label():
    bi = _input(sections=["weaknesses", "investment_perspective"])
    assert uncovered_sections("Nothing relevant here.", bi) == ["Weaknesses", "Investment Perspective"]
    # Planner explicitly assigned the id -> covered even with a creative heading.
    assert uncovered_sections("", bi, outline_text='{"brief": "covers weaknesses"}') == ["Investment Perspective"]


def test_editorial_integrity_reports_each_failure_in_plain_language():
    bi = _input(sections=["weaknesses"])
    content = BlogContent(title="T", seo_keywords=[], sections=[BlogSection(heading="H", body_markdown="x")])
    score, issues = compute_editorial_integrity(content, "Nestled here, prices rose 14%.", bi)
    assert score == 20.0  # 4 of 5 checks failed: sections, CTA, figures, phrasing
    text = " ".join(issues)
    assert "call to action" in text and "14%" in text and "nestled" in text and "Weaknesses" in text


def test_clean_post_scores_full_marks():
    bi = _input(sections=["weaknesses"], local_facts=["2 km to metro"])
    content = BlogContent(title="T", cta_markdown="Book a visit.", seo_keywords=["noida homes"])
    score, issues = compute_editorial_integrity(content, "The weaknesses: the metro is 2 km away.", bi)
    assert score == 100.0 and issues == []


def test_brief_warns_writer_when_no_facts_are_given():
    assert "do NOT state any specific numbers" in brief.local_facts_block(_input())


def test_cta_instruction_never_invites_invented_contacts():
    assert "without inventing" in brief.cta_instruction(_input())
    assert "do not repeat them" in brief.cta_instruction(_input(cta_phone="+91 98100 00000"))


def test_old_blog_inputs_still_load():
    """Existing blog_input.json files use key_points and no sections."""
    bi = BlogInput.model_validate({"blog_id": "B", "topic": "T", "key_points": ["a"], "tone": "friendly"})
    assert bi.sections == [] and bi.key_points == ["a"]
