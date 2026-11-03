"""Fact Check Agent — does the draft only say what we can support?

Research is worth nothing on its own. A model handed a pack of real
distances will still write "just 15 minutes from the airport" if 15
sounds better than the 35 it was given, and nothing downstream would
notice. This is the step that notices.

HOW IT WORKS, AND WHY IN THAT ORDER

Deterministically first, then a model. Code pulls out every checkable
claim — anything with a number, a distance, a date, a price — because
finding them is pattern matching and a model asked to find them will miss
some and invent others. Then each claim is matched against the evidence:
the author's own facts, and the research pack. Anything that matches is
verified and never reaches the model at all.

Only what is left goes to an LLM, and only for the judgement code cannot
make: whether "a short drive from the ring road" is supported by "4.2 km
to Sardar Patel Ring Road". That is a language question. "Is 35 in this
list of numbers" is not, and paying a model to answer it would be both
slower and less reliable.

WHAT IS NOT A FAILURE

Opinions. "Shela suits families who want space over a short commute" is
a view, not a claim, and flagging it would train the writer out of having
one — which would make the posts worse, not more accurate. Only
falsifiable statements are checked.

WHAT HAPPENS TO WHAT IT FINDS

The unsupported claims go into the evaluation's issues, which lowers
editorial integrity, which trips the existing quality gate, which sends
the post back to the writer WITH THE SPECIFIC CLAIMS NAMED. The revision
loop already existed; before this it was blind — the writer was asked to
try again with no idea what was wrong.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from string import Template
from typing import Optional

from src.agents.blog import llm
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.data.schemas import BlogContent, BlogInput, FactCheckResult, ResearchPack
from src.models.gateway import call_with_fallback

logger = logging.getLogger(__name__)

# A sentence is worth checking if it asserts something falsifiable. In
# practice that means a number with units, a year, a price, or a distance.
_CHECKABLE = re.compile(
    r"""(
        \d[\d,.]*\s*(?:km|kms|kilometre|kilometres|kilometer|kilometers|m|metre|metres|
                       min|mins|minute|minutes|hour|hours|hrs|
                       %|per\s*cent|percent|
                       sq\.?\s*ft|sqft|square\s+feet|acre|acres|
                       cr|crore|crores|lakh|lakhs|bhk)
      | (?:₹|rs\.?|inr|\$)\s*\d
      | \b(?:19|20)\d{2}\b
    )""",
    re.I | re.X)

# Numbers that assert nothing: list positions, a "3 BHK" in the topic
# itself, section numbering.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_CLAIMS_TO_JUDGE = 12


def run(blog_input: BlogInput, content: BlogContent,
        research: Optional[ResearchPack] = None) -> FactCheckResult:
    """Which claims in the finished draft the evidence actually supports."""
    claims = extract_claims(content)
    if not claims:
        return FactCheckResult(notes=["The post makes no specific factual claims to check."])

    evidence = _evidence(blog_input, research)
    verified, unknown = [], []
    for claim in claims:
        (verified if _supported_by_the_numbers(claim, evidence) else unknown).append(claim)

    if not unknown:
        return FactCheckResult(verified=verified)

    # Only the genuinely ambiguous ones cost a model call.
    judged = _judge(unknown[:MAX_CLAIMS_TO_JUDGE], evidence)
    if judged is None:
        return FactCheckResult(
            verified=verified, unsupported=unknown,
            notes=["Fact checking could not reach the model, so unverified claims are "
                   "listed conservatively."])
    verified.extend(judged["supported"])
    return FactCheckResult(verified=verified, unsupported=judged["unsupported"])


# ---------------------------------------------------------------------------

def extract_claims(content: BlogContent) -> list[str]:
    """Every sentence in the post that asserts something checkable."""
    body = [content.intro_markdown, content.conclusion_markdown, content.cta_markdown]
    body += [s.body_markdown for s in content.sections]
    body += list(getattr(content, "key_takeaways", []) or [])
    body += [f.answer_markdown for f in getattr(content, "faqs", []) or []]
    claims: list[str] = []
    seen: set[str] = set()
    for block in body:
        for sentence in _SENTENCE_SPLIT.split(block or ""):
            text = re.sub(r"[*_`#>\[\]|]", " ", sentence)
            text = " ".join(text.split()).strip(" -")
            if not text or not _CHECKABLE.search(text):
                continue
            key = text.lower()
            if key not in seen:
                seen.add(key)
                claims.append(text)
    return claims


def evidence_for(blog_input: BlogInput, research: Optional[ResearchPack]) -> dict:
    """Everything the post is allowed to assert, as text and as numbers.

    Public: the Editor (which removes untraceable figures) and the rubric
    (which counts them) use exactly this, so "traceable" means one thing
    across the pipeline."""
    return _evidence(blog_input, research)


def _brief_text(blog_input: BlogInput) -> str:
    """Every part of the brief a figure may legitimately come from — the
    topic ("Sector 150"), the points, the project name — not only facts."""
    parts = [blog_input.topic, *blog_input.key_points, *getattr(blog_input, "custom_sections", []),
             getattr(blog_input, "location", ""), getattr(blog_input, "project_name", ""),
             getattr(blog_input, "brand_name", ""), getattr(blog_input, "brand_description", ""),
             getattr(blog_input, "audience", ""), blog_input.cta_text or "",
             getattr(blog_input, "primary_keyword", ""), *getattr(blog_input, "secondary_keywords", [])]
    return " ".join(p for p in parts if p)


def _evidence(blog_input: BlogInput, research: Optional[ResearchPack]) -> dict:
    """Everything the post is allowed to assert, as text and as numbers."""
    lines = [f"- (author's own fact) {f}" for f in blog_input.local_facts if f.strip()]
    numbers: set[str] = set()
    for fact in blog_input.local_facts:
        numbers.update(_numbers_in(fact))
    numbers.update(_numbers_in(_brief_text(blog_input)))
    # The current year is context ("as of 2026"), not a claim.
    numbers.add(str(date.today().year))

    if research is not None:
        for place in research.nearby:
            lines.append(f"- {place.name} is about {place.distance_km} km away (straight line)")
            numbers.add(_normalise(place.distance_km))
        for journey in research.journeys:
            lines.append(f"- {journey.destination}: {journey.distance_km} km by road, "
                         f"about {journey.free_flow_minutes} minutes in clear traffic")
            numbers.add(_normalise(journey.distance_km))
            numbers.add(_normalise(journey.free_flow_minutes))
        if research.background:
            lines.append(f"- (background) {research.background}")
            numbers.update(_numbers_in(research.background))
        for item in research.news:
            lines.append(f"- (headline, {item.source}) {item.title}")
            numbers.update(_numbers_in(item.title))
        for ref in getattr(research, "official_sources", []) or []:
            if ref.note:
                lines.append(f"- (official source: {ref.title}) {ref.note}")
                numbers.update(_numbers_in(ref.note))
    return {"text": "\n".join(lines) or "(nothing was provided or found)", "numbers": numbers}


def numbers_in(text: str) -> set[str]:
    return _numbers_in(text)


def _numbers_in(text: str) -> set[str]:
    return {_normalise(n) for n in re.findall(r"\d[\d,]*\.?\d*", text or "")}


def _normalise(value) -> str:
    """So that 18, 18.0 and 18,000 -> 18000 compare as the same figure."""
    try:
        number = float(str(value).replace(",", "").rstrip("."))
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number == int(number) else str(number)


# A FIGURE: a number that asserts something — money, a measurement with a
# unit, a percentage, a year, or any number of three or more digits.
# Small bare counts ("3 steps", "2 BHK", "Phase 2") are not figures: they
# rarely carry a factual claim, and treating them as one would strip
# harmless structure from the post.
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
FIGURE = re.compile(
    r"(?:₹|rs\.?|inr|\$|usd|aed)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:[-–—]|to)\s?\d[\d,]*(?:\.\d+)?)?"
    r"(?:\s?(?:lakh|lakhs|lac|crore|crores|cr|k|million|mn|bn|billion))?"
    r"|\b\d[\d,]*(?:\.\d+)?(?:\s?(?:[-–—]|to)\s?\d[\d,]*(?:\.\d+)?)?[\s-]?"
    r"(?:%|per\s?cent|percent|lakh|lakhs|lac|crore|crores|cr\b|km\b|kms\b|kilomet\w+|metres?\b|meters?\b|"
    r"m\b|min\b|mins\b|minutes?\b|hours?\b|hrs\b|sq\.?\s?ft\b|sqft\b|square\s+(?:feet|foot|met\w+)|"
    r"sq\.?\s?m\b|acres?\b|hectares?\b|years?\b|yrs\b|months?\b|storeys?\b|floors\b|towers\b|"
    r"units\b|homes\b|apartments\b|flats\b|times\b|x\b)"
    r"|(?:" + _MONTHS + r"\s+)?\b(?:19|20)\d{2}\b"
    r"|\b\d+\.\d+\b"
    r"|\b\d[\d,]{2,}\b",
    re.I)


def figures_in(text: str) -> list[str]:
    return [m.group(0).strip() for m in FIGURE.finditer(text or "")]


def untraceable_figures(text: str, evidence: dict) -> list[str]:
    """Figures in `text` with at least one number the evidence lacks."""
    out = []
    for figure in figures_in(text):
        numbers = _numbers_in(figure)
        if numbers and not numbers.issubset(evidence["numbers"]):
            out.append(figure)
    return out


def _supported_by_the_numbers(claim: str, evidence: dict) -> bool:
    """True when every figure in the claim appears in the evidence.

    A cheap first pass that settles most claims without a model call. It is
    deliberately strict about what it will clear: all figures must match,
    so a sentence that mixes one real distance with one invented one is
    NOT waved through on the strength of the real one."""
    figures = _numbers_in(claim)
    if not figures:
        return False
    return figures.issubset(evidence["numbers"])


def _judge(claims: list[str], evidence: dict) -> Optional[dict]:
    """The language judgement code cannot make. None if the model is
    unreachable — the caller then reports the claims as unverified, which
    is the safe direction to be wrong in."""
    template = load_yaml_config("blog/prompts.yaml").get("fact_check")
    if not template:
        return None
    prompt = Template(template).substitute(
        evidence=evidence["text"],
        claims="\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1)),
    )
    try:
        result = llm.generate_json(ModelRole.BLOG_FACT_CHECK, prompt, call_with_fallback)
    except Exception as exc:  # noqa: BLE001 - fact checking must not break a post
        logger.warning("Fact check model call failed: %s", exc)
        return None

    supported_indexes = {int(i) for i in (result.get("supported") or []) if str(i).isdigit()}
    supported = [c for i, c in enumerate(claims, start=1) if i in supported_indexes]
    unsupported = [c for i, c in enumerate(claims, start=1) if i not in supported_indexes]
    return {"supported": supported, "unsupported": unsupported}


def as_revision_note(result: FactCheckResult) -> str:
    """What to tell the writer when the post comes back for another pass.

    Naming the sentences is the whole point. The revision loop existed
    before this and was blind: "try again, the quality gate failed" gets a
    reworded post with the same invented numbers in it."""
    if result.ok:
        return ""
    lines = ["These sentences state something the evidence does not support. Either remove "
             "the figure, attribute it, or rewrite the sentence without it:"]
    lines += [f'- "{claim}"' for claim in result.unsupported]
    return "\n".join(lines)
