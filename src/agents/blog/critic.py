"""Critic Agent — scores the post on the five-parameter rubric
(configs/blog/evaluation.yaml) and writes the feedback the revision loop
sends back to the Writer.

Deterministic measurements (src/evaluation/blog_metrics.py) carry most of
each parameter's score; the LLM critic, when reachable, scores the same
five parameters — the prompt lists them, with their weights, from the same
config — gives a one-line reason for each and a note for each weak one,
blended in at `llm_blend`. With no model the rubric still scores every
parameter, which is what lets the quality gate and the UI work offline.

The fact check's and the Editor's findings arrive here rather than being
acted on separately, because this is where the quality gate is: one loop,
not two.
"""
from __future__ import annotations

import logging
from string import Template
from typing import Optional

from src.agents.blog import article, brief, editor, llm
from src.agents.blog import fact_check as fact_check_mod
from src.core.config import load_yaml_config
from src.core.types import ModelRole
from src.data.schemas import (
    BlogContent, BlogEvaluationResult, BlogInput, BlogOutline, BlogValidationReport, FactCheckResult, ResearchPack,
)
from src.evaluation import blog_metrics, blog_rubric, blog_scoring
from src.evaluation.blog_deterministic import compute_editorial_integrity, compute_structural_completeness
from src.models.gateway import call_with_fallback

logger = logging.getLogger(__name__)


def _full_markdown(content: BlogContent) -> str:
    return article.full_markdown(content)


def _llm_judgement(blog_input: BlogInput, content: BlogContent, post_markdown: str) -> Optional[dict]:
    try:
        spec = blog_rubric.dimension_spec()
        listing = "\n".join(f"    {d['name']:<22} ({d['weight']:.0%}) {d['label']}: {d['description']}"
                             for d in spec)
        zeros = ", ".join(f'"{d["name"]}": 0' for d in spec)
        reasons = ", ".join(f'"{d["name"]}": "one line"' for d in spec)
        prompt = Template(load_yaml_config("blog/prompts.yaml")["critic"]).substitute(
            dimensions=listing, dimension_scores=zeros, dimension_reasons=reasons,
            topic=blog_input.topic,
            tone=brief.tone(blog_input),
            requested_sections=brief.requested_sections_block(blog_input),
            local_facts=brief.local_facts_block(blog_input),
            brief_profile=brief.brief_profile_block(blog_input),
            post_markdown=post_markdown,
        )
        result = llm.generate_json(ModelRole.BLOG_CRITIC, prompt, call_with_fallback)
        return result if isinstance(result, dict) else None
    except Exception as exc:  # noqa: BLE001 - the rubric scores without a model
        logger.warning("Critic model unavailable; scoring deterministically: %s", exc)
        return None


# Parameters a measurement can check only in part: without a reader model
# they are capped (and say so) rather than claim more than was checked.
JUDGEMENT_DIMENSIONS = ("content_quality", "engagement")
MEASURED_CAP = 82.0


def run(blog_input: BlogInput, content: BlogContent, outline: Optional[BlogOutline] = None,
        fact_check: Optional[FactCheckResult] = None, research: Optional[ResearchPack] = None,
        validation: Optional[BlogValidationReport] = None) -> BlogEvaluationResult:
    """Score the finished post."""
    post_markdown = _full_markdown(content)
    judged = _llm_judgement(blog_input, content, post_markdown)
    outline_text = outline.model_dump_json() if outline else ""
    evidence = fact_check_mod.evidence_for(blog_input, research)
    measurement = blog_metrics.measure(content, blog_input, research=research, evidence=evidence,
                                       outline_text=outline_text)
    ctx = brief.context(blog_input)

    names = [d["name"] for d in blog_rubric.dimension_spec()]
    llm_scores = (judged or {}).get("dimensions") if judged else None
    if judged and not isinstance(llm_scores, dict):
        llm_scores = {k: judged[k] for k in names if k in judged} or None
    notes = (judged or {}).get("notes") if judged else None
    reasons = (judged or {}).get("reasons") if judged else None
    dimensions = blog_rubric.score_dimensions(
        measurement, llm_scores=llm_scores,
        has_research_places=bool(research and research.nearby),
        location_known=ctx.location.known, brand_given=bool(blog_input.brand_name),
        llm_reasons=reasons if isinstance(reasons, dict) else None)
    if not judged:
        # Measurements can show a post is well built; they cannot show it
        # reads well or persuades. Without a reader model those dimensions
        # are capped, and say why, rather than claim more than was checked.
        for d in dimensions:
            if d.name in JUDGEMENT_DIMENSIONS and d.score > MEASURED_CAP:
                d.score = MEASURED_CAP
                d.passed = d.score >= d.threshold
                d.findings.insert(0, "Measured only: no model reviewed the writing for this parameter.")
                if not d.findings[1:]:
                    d.summary = ("No problems measured; capped at "
                                 f"{MEASURED_CAP:.0f} because no model reviewed the writing.")
    by_name = {d.name: d for d in dimensions}

    # Legacy fields, still shown by older UI code and stored rows.
    structural_completeness = compute_structural_completeness(content, post_markdown, blog_input)
    editorial_integrity, issues = compute_editorial_integrity(
        content, post_markdown, blog_input, outline_text, allowed_numbers=evidence["numbers"])
    grounding = by_name["content_quality"].score
    writing_quality = round((by_name["structure_readability"].score + by_name["engagement"].score) / 2, 1)
    if judged:
        if isinstance(judged.get("grounding"), (int, float)):
            grounding = round(0.5 * grounding + 0.5 * float(judged["grounding"]), 1)
        if isinstance(judged.get("writing_quality"), (int, float)):
            writing_quality = round(0.5 * writing_quality + 0.5 * float(judged["writing_quality"]), 1)

    if fact_check is not None and fact_check.unsupported:
        issues.append(f"{len(fact_check.unsupported)} statement(s) in the draft were not supported by the "
                      "evidence and were removed or need attribution.")
    issues.extend(n for n in (fact_check.notes if fact_check else []) if "no specific factual" not in n)
    if validation is not None:
        issues.extend(i.message for i in validation.issues if i.severity in ("error", "warning"))
    for d in dimensions:
        if not d.passed:
            issues.append(f"{d.label} below target ({d.score:.0f}/{d.threshold:.0f}): "
                          + (d.findings[0] if d.findings else "see the evaluation breakdown"))

    feedback = blog_rubric.feedback(dimensions, notes if isinstance(notes, dict) else None)
    if validation is not None:
        feedback += editor.as_revision_note(validation)

    evaluation = BlogEvaluationResult(
        grounding=grounding,
        writing_quality=writing_quality,
        structural_completeness=structural_completeness,
        editorial_integrity=editorial_integrity,
        issues=list(dict.fromkeys(issues)),
        dimensions=dimensions,
        metrics={k: float(v) for k, v in measurement.values.items()},
        feedback=feedback,
        critic_mode="llm+deterministic" if judged else "deterministic",
    )
    evaluation.quality_score = blog_scoring.compute_quality_score(evaluation)
    evaluation.gate_result = blog_scoring.apply_quality_gate(evaluation)
    if validation is not None and validation.blocking:
        from src.core.types import QualityGateResult

        evaluation.gate_result = QualityGateResult.REVISE
    return evaluation
