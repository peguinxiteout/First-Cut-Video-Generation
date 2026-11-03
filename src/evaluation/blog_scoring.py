"""
Blog Quality Score and quality gate.

The score is the weighted sum of the five rubric parameters
(configs/blog/evaluation.yaml, src/evaluation/blog_rubric.py) when they
are present — which they are for every evaluation the critic produces
now. Evaluations stored before the rubric existed have no dimensions and
are scored the old way, from the four legacy fields and
configs/evaluation.yaml `blog_weights`, so reloading an old record does
not change its score.

The gate asks for a revision when the overall score is under
`gate.pass_score` OR any dimension is under its own threshold: a post
that is excellent on average but invents figures is not publishable, and
averaging must not hide that.
"""
from __future__ import annotations

from src.agents.blog import blog_config
from src.core.config import load_yaml_config
from src.core.types import QualityGateResult
from src.data.schemas import BlogEvaluationResult
from src.evaluation.tiers import quality_tier  # noqa: F401 - re-exported for convenience

_WEIGHTS = load_yaml_config("evaluation.yaml")["blog_weights"]


def compute_quality_score(evaluation: BlogEvaluationResult) -> float:
    if evaluation.dimensions:
        from src.evaluation.blog_rubric import overall

        return overall(evaluation.dimensions)
    total = (
        evaluation.grounding * _WEIGHTS["grounding"]
        + evaluation.writing_quality * _WEIGHTS["writing_quality"]
        + evaluation.structural_completeness * _WEIGHTS["structural_completeness"]
        + evaluation.editorial_integrity * _WEIGHTS.get("editorial_integrity", 0)
    )
    return round(total, 2)


def apply_quality_gate(evaluation: BlogEvaluationResult) -> QualityGateResult:
    gate = blog_config.rubric().get("gate", {})
    score = evaluation.quality_score
    if evaluation.dimensions:
        if score >= gate.get("pass_score", 78) and all(d.passed for d in evaluation.dimensions):
            return QualityGateResult.PASS
        return QualityGateResult.REVISE
    app = load_yaml_config("app.yaml")["quality_gate"]
    if score >= app["revision_threshold"]:
        return QualityGateResult.PASS
    return QualityGateResult.REVISE


def max_revisions() -> int:
    return int(blog_config.rubric().get("gate", {}).get("max_revisions", 2))


def min_improvement() -> float:
    return float(blog_config.rubric().get("gate", {}).get("min_improvement", 0.5))
