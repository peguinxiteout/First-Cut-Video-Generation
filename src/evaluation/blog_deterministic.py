"""
Deterministic checks for blog posts — feed the `structural_completeness`
dimension of BlogEvaluationResult (see blog_scoring.py). SEO fields
(title/meta description/keywords) are checked here rather than by an LLM
judgment call, since their presence and shape are objectively verifiable,
not a matter of taste.
"""
from __future__ import annotations

from src.data.schemas import BlogContent, BlogInput

_MIN_META_DESCRIPTION_LEN = 50
_MAX_META_DESCRIPTION_LEN = 160
_MIN_KEYWORDS = 3
_MIN_SECTIONS = 2
_WORD_COUNT_TOLERANCE = 0.35  # within 35% of target counts as "near enough"


def _title_present(content: BlogContent) -> bool:
    return bool(content.title.strip())


def _meta_description_well_formed(content: BlogContent) -> bool:
    length = len(content.meta_description.strip())
    return _MIN_META_DESCRIPTION_LEN <= length <= _MAX_META_DESCRIPTION_LEN


def _keywords_present(content: BlogContent) -> bool:
    return len(content.seo_keywords) >= _MIN_KEYWORDS


def _sections_adequate(content: BlogContent) -> bool:
    if len(content.sections) < _MIN_SECTIONS:
        return False
    return all(s.body_markdown.strip() for s in content.sections)


def _word_count_near_target(post_markdown: str, target_word_count: int | None) -> bool:
    if not target_word_count:
        return True  # no target given — nothing to check
    actual = len(post_markdown.split())
    lower = target_word_count * (1 - _WORD_COUNT_TOLERANCE)
    upper = target_word_count * (1 + _WORD_COUNT_TOLERANCE)
    return lower <= actual <= upper


def compute_structural_completeness(
    content: BlogContent, post_markdown: str, blog_input: BlogInput
) -> float:
    checks = [
        _title_present(content),
        _meta_description_well_formed(content),
        _keywords_present(content),
        _sections_adequate(content),
        _word_count_near_target(post_markdown, blog_input.target_word_count),
    ]
    return round(100.0 * sum(checks) / len(checks), 2)


# ---------------------------------------------------------------------------
# Editorial integrity — the specific failures found in the original-vs-
# generated blog evaluation, checked in code rather than left to an LLM:
#   * formulaic/AI phrasing          -> banned stock phrases
#   * keyword stuffing (original)    -> repeated search phrases
#   * missing call to action (both)  -> CTA present
#   * vague vs invented specifics    -> every figure must trace to the facts
#   * user-selected sections         -> each one covered
# ---------------------------------------------------------------------------
import re as _re

BANNED_PHRASES = [
    "nestled", "in today's fast-paced world", "look no further", "unlock", "elevate your lifestyle",
    "a testament to", "in conclusion", "dive into", "game-changer", "game changer", "seamlessly",
    "boasts", "a haven", "the perfect blend",
]

# Figures that carry factual weight: percentages, currency, figures with
# units, and any number of 3+ digits. Small bare counts ("3 bedrooms")
# are left alone — they rarely carry a factual claim on their own.
_FIGURE = _re.compile(
    r"(?:₹|rs\.?|inr|\$)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:lakh|lakhs|crore|crores|cr|k|m))?"
    r"|\d[\d,]*(?:\.\d+)?\s?(?:%|percent|per cent|lakh|lakhs|crore|crores|cr|km|kms|kilometres|"
    r"kilometers|minutes|mins|min|sq\.?\s?ft|sqft|square feet|acres|acre|years|yrs)"
    r"|\b\d{3,}[\d,]*\b",
    _re.IGNORECASE,
)


def _digits(text: str) -> set[str]:
    return {m.replace(",", "") for m in _re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def unsupported_figures(post_markdown: str, blog_input: BlogInput,
                        allowed_numbers: set[str] | None = None) -> list[str]:
    """Figures in the post whose numbers never appear in the author's
    facts, topic, or points — i.e. likely invented. `allowed_numbers`
    adds the research evidence (see fact_check.evidence_for), so a
    measured distance is not reported as invented."""
    allowed = _digits(" ".join(blog_input.local_facts + blog_input.key_points + [blog_input.topic]))
    if allowed_numbers:
        allowed |= set(allowed_numbers)
    flagged = []
    for match in _FIGURE.finditer(post_markdown):
        figure = match.group(0)
        numbers = _digits(figure)
        if numbers and not numbers & allowed:
            flagged.append(figure.strip())
    return sorted(set(flagged))


def stuffed_keywords(post_markdown: str, keywords: list[str], max_repeats: int = 3) -> list[str]:
    """Multi-word search phrases repeated more often than natural prose
    would — the original blog's 'flats for sale Electronic City Phase 1'
    pattern."""
    text = post_markdown.lower()
    stuffed = []
    for kw in keywords:
        phrase = kw.strip().lower()
        if len(phrase.split()) >= 3 and text.count(phrase) > max_repeats:
            stuffed.append(kw)
    return stuffed


def banned_phrases_used(post_markdown: str) -> list[str]:
    text = post_markdown.lower()
    return [p for p in BANNED_PHRASES if _re.search(r"\b" + _re.escape(p) + r"\b", text)]


# Words too common to prove anything. A custom section matched on "the",
# "with" and "property" would be judged covered by any post about property.
_STOPWORDS = frozenset({
    "about", "after", "also", "area", "around", "because", "been", "being", "best",
    "between", "both", "could", "does", "during", "each", "from", "have", "here",
    "how", "into", "its", "just", "like", "made", "make", "many", "more", "most",
    "much", "must", "near", "next", "only", "other", "our", "over", "own", "same",
    "should", "since", "some", "such", "than", "that", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "under", "until", "very",
    "what", "when", "where", "which", "while", "will", "with", "would", "your",
    "blog", "post", "article", "section", "write", "include", "property", "real",
    "estate", "home", "house", "buy", "buyer", "buyers", "project", "projects",
})


def _distinctive_words(text: str) -> list[str]:
    return [w for w in _re.findall(r"[a-z]+", text.lower())
            if len(w) > 3 and w not in _STOPWORDS]


def _stem(word: str) -> str:
    """A crude prefix, enough to match a word against its own inflections.

    The writer chooses its own headings and phrasing, so "becoming" in the
    brief comes back as "become" in the post. Matching whole words only
    failed every well-written rewording, which would have made the check
    punish exactly the behaviour it wants."""
    return word[:5] if len(word) > 5 else word


def uncovered_sections(post_markdown: str, blog_input: BlogInput, outline_text: str = "") -> list[str]:
    """Which requested sections the finished post does not visibly cover.

    A CATALOGUE section counts as covered if the planner assigned its id to
    a section, or its label's words appear in the post.

    A CUSTOM section — a sentence the person typed — has no id and no
    label, so it is matched on its own distinctive words instead: strip
    the ones common to any property article ("property", "buyers",
    "area") and require most of what is left to appear. Requiring ALL of
    them would fail on any sensible rewording, and requiring ANY of them
    would pass on the word "investment" alone, which is why it is a
    proportion."""
    from src.agents.blog.brief import custom_sections, section_catalog

    catalog = section_catalog()
    haystack = (post_markdown + " " + outline_text).lower()
    missing = []
    for sid in blog_input.sections:
        label = catalog.get(sid, {}).get("label", sid)
        label_words = [w for w in _re.findall(r"[a-z]+", label.lower()) if len(w) > 3]
        if sid.lower() in haystack:
            continue
        if label_words and all(w in haystack for w in label_words):
            continue
        missing.append(label)

    for text in custom_sections(blog_input):
        words = _distinctive_words(text)
        if not words:
            continue        # nothing specific enough to check for
        stems = [_stem(w) for w in words]
        hits = sum(1 for stem in stems if stem in haystack)
        if hits / len(stems) < 0.6:
            missing.append(f'your own section: "{_shorten(text)}"')
    return missing


def _shorten(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "\u2026"


def compute_editorial_integrity(
    content: BlogContent, post_markdown: str, blog_input: BlogInput, outline_text: str = "",
    allowed_numbers: set[str] | None = None,
) -> tuple[float, list[str]]:
    issues: list[str] = []
    missing = uncovered_sections(post_markdown, blog_input, outline_text)
    if missing:
        issues.append("Requested sections not clearly covered: " + ", ".join(missing))
    if not (content.cta_markdown or "").strip():
        issues.append("No call to action.")
    stuffed = stuffed_keywords(post_markdown, content.seo_keywords)
    if stuffed:
        issues.append("Keyword stuffing: " + ", ".join(stuffed))
    invented = unsupported_figures(post_markdown, blog_input, allowed_numbers)
    if invented:
        issues.append("Figures not found in the provided facts (possibly invented): " + ", ".join(invented[:6]))
    banned = banned_phrases_used(post_markdown)
    if banned:
        issues.append("Stock AI phrasing: " + ", ".join(banned))
    checks = 5
    score = 100.0 * (checks - len(issues)) / checks
    return round(score, 2), issues
