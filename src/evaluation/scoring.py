"""
Brochure Quality Score: 3 dimensions.

Why these three, why no "completeness": a prior version had a 4th
dimension — a deterministic "completeness" check (PDF exists, hero image
selected, required sections present, copy fields non-empty) — that
partially contributed to the weighted Quality Score. In practice this
inflated the score: a brochure with mediocre writing or an awkward layout
still scored well on "completeness" just because the file was structurally
present, pulling the overall number up even when the actual content/design
quality didn't deserve it. Whether the PDF rendered at all is now a binary
gate BEFORE the Critic ever runs (see is_pdf_render_valid() in
src/rendering/renderer.py, used by workflows/brochure.py's node_render) —
a generation with a broken/missing PDF fails outright rather than reaching
the Critic with a partial completeness credit. So all three remaining
dimensions are genuine quality judgments, not structural presence checks:
factual_consistency (grounding), visual_design_quality (the VLM's holistic
read of the rendered page), and content_quality (the writing itself).

Why "factual_consistency" not "factual_accuracy": in Content Plan (manual)
mode, the copy is verbatim, user-authored text — it isn't LLM-generated, so
grading it against "is every sentence traceable to property.json" would
unfairly penalize rich manually-authored content that simply wasn't
pre-declared in the (necessarily sparse) fact ledger. Grading for
CONTRADICTION instead of exhaustive traceability is the correct check for
both modes: it still catches real errors (e.g. copy says "4 BHK" but
property.json says "3 BHK") without penalizing legitimate detail.
"""
from __future__ import annotations

from src.core.config import load_yaml_config, settings
from src.core.types import QualityGateResult
from src.data.schemas import EvaluationResult
from src.evaluation.tiers import quality_tier  # noqa: F401 - re-exported for convenience

_WEIGHTS = load_yaml_config("evaluation.yaml")["brochure_weights"]
_LAYOUT_WEIGHT = float(load_yaml_config("evaluation.yaml").get("brochure_layout_weight", 0.10))
_APP_CFG = load_yaml_config("app.yaml")["quality_gate"]


def compute_quality_score(evaluation: EvaluationResult) -> float:
    total = (
        evaluation.factual_consistency * _WEIGHTS["factual_consistency"]
        + evaluation.visual_design_quality * _WEIGHTS["visual_design_quality"]
        + evaluation.content_quality * _WEIGHTS["content_quality"]
    )
    if evaluation.layout_integrity is not None:
        w = _LAYOUT_WEIGHT
        total = (1 - w) * total + w * evaluation.layout_integrity
    return round(total, 2)


# Fixed scores per factual_consistency_status category (see critic_combined's
# DIMENSION 1 in configs/brochure/prompts.yaml) — a deterministic lookup, not
# a number the model picks itself. This exists because letting an LLM
# self-report a fine-grained 0-100 number for what's fundamentally a
# near-binary judgment (contradicted or not) produced real, reported
# inconsistency: reasoning that confidently found zero contradictions was
# still sometimes paired with a self-picked score like 78 or 95 instead of
# 100. Asking the model for a category instead, and deriving the score from
# that category in code, makes the score a function OF the reasoning rather
# than a second, independently-generated number that can drift from it.
_FACTUAL_CONSISTENCY_SCORES = {
    "consistent": 100.0,
    "minor_issue": 85.0,
    "fabrication": 55.0,
    "contradiction": 20.0,
}
_FACTUAL_CONSISTENCY_FALLBACK = 70.0  # used only if the model returns an unrecognized/missing status


def factual_consistency_score_from_status(status: str) -> float:
    """Maps the Critic's factual_consistency_status classification to its
    fixed score. An unrecognized value (the model didn't follow the
    instructions, or the field is missing) falls back to a deliberately
    middling score rather than either extreme — this should be rare, and
    is a "we couldn't classify this" signal rather than a real judgment
    either way, so it shouldn't read as confidently good or bad."""
    return _FACTUAL_CONSISTENCY_SCORES.get((status or "").strip().lower(), _FACTUAL_CONSISTENCY_FALLBACK)


def apply_quality_gate(evaluation: EvaluationResult) -> QualityGateResult:
    score = evaluation.quality_score
    if score >= _APP_CFG["auto_pass_threshold"]:
        return QualityGateResult.PASS
    if score >= _APP_CFG["revision_threshold"]:
        return QualityGateResult.PASS  # good enough to send to human review, not auto-fail
    return QualityGateResult.REVISE
