"""
Shared human-readable quality tier, layered on top of the numeric Quality
Score — used by both brochure and blog evaluation so scores are never shown
as a bare number without context. Deliberately strict: "Excellent" is
reserved for genuinely flawless-or-near-flawless output, not just "pretty
good" (see the Critic prompts' scoring rubric for the matching anchors
applied when the LLM/VLM judges assign their raw dimension scores).
"""
from __future__ import annotations

TIER_EXCELLENT = "Excellent"
TIER_VERY_GOOD = "Very Good"
TIER_GOOD = "Good"
TIER_NEEDS_IMPROVEMENT = "Needs Improvement"


def quality_tier(score: float) -> str:
    if score >= 95:
        return TIER_EXCELLENT
    if score >= 85:
        return TIER_VERY_GOOD
    if score >= 70:
        return TIER_GOOD
    return TIER_NEEDS_IMPROVEMENT
