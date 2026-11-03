from __future__ import annotations

from src.core.types import QualityGateResult
from src.data.schemas import EvaluationResult
from src.evaluation.scoring import apply_quality_gate, compute_quality_score
from src.evaluation.tiers import quality_tier


def test_compute_quality_score_is_weighted_average_of_three_dimensions():
    evaluation = EvaluationResult(factual_consistency=100, visual_design_quality=100, content_quality=100)
    assert compute_quality_score(evaluation) == 100.0


def test_compute_quality_score_weights_sum_correctly():
    """A uniform 80 across all dimensions must produce exactly 80,
    regardless of individual weight values — a basic sanity check that the
    weights in configs/evaluation.yaml actually sum to 1.0."""
    evaluation = EvaluationResult(factual_consistency=80, visual_design_quality=80, content_quality=80)
    assert compute_quality_score(evaluation) == 80.0


def test_no_completeness_field_on_evaluation_result():
    """Regression test: 'completeness' was removed as an evaluation
    dimension entirely — it was a structural pass/fail check that inflated
    the overall score even when actual content/design quality was
    mediocre. Whether the PDF rendered validly is now a hard gate before
    the Critic runs at all (see test_renderer.py's is_pdf_render_valid
    tests), not a partial-credit contributor to the Quality Score."""
    assert "completeness" not in EvaluationResult.model_fields


def test_quality_gate_pass_above_threshold():
    evaluation = EvaluationResult(quality_score=95)
    assert apply_quality_gate(evaluation) == QualityGateResult.PASS


def test_quality_gate_revise_below_threshold():
    evaluation = EvaluationResult(quality_score=50)
    assert apply_quality_gate(evaluation) == QualityGateResult.REVISE


def test_quality_tier_labels():
    assert quality_tier(97) == "Excellent"
    assert quality_tier(90) == "Very Good"
    assert quality_tier(75) == "Good"
    assert quality_tier(40) == "Needs Improvement"
    # Boundary checks — exact threshold values belong to the HIGHER tier.
    assert quality_tier(95) == "Excellent"
    assert quality_tier(85) == "Very Good"
    assert quality_tier(70) == "Good"
    assert quality_tier(69.9) == "Needs Improvement"
