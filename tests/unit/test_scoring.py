from __future__ import annotations

from src.evaluation.scoring import factual_consistency_score_from_status


def test_consistent_status_scores_exactly_100():
    """The reported bug: reasoning confidently finding zero contradictions
    was still sometimes paired with a self-picked score like 78 or 95
    instead of 100. The score is now a deterministic function of the
    status, not a second number the model picks independently — so
    "consistent" must always be exactly 100, with no room to drift."""
    assert factual_consistency_score_from_status("consistent") == 100.0


def test_minor_issue_scores_lower_than_consistent_but_still_high():
    assert factual_consistency_score_from_status("minor_issue") == 85.0


def test_fabrication_scores_in_the_middle():
    assert factual_consistency_score_from_status("fabrication") == 55.0


def test_contradiction_scores_low():
    assert factual_consistency_score_from_status("contradiction") == 20.0


def test_scores_are_strictly_ordered_by_severity():
    consistent = factual_consistency_score_from_status("consistent")
    minor = factual_consistency_score_from_status("minor_issue")
    fabrication = factual_consistency_score_from_status("fabrication")
    contradiction = factual_consistency_score_from_status("contradiction")
    assert consistent > minor > fabrication > contradiction


def test_unrecognized_status_falls_back_to_a_middling_score_not_an_extreme():
    """If the model doesn't follow instructions (missing/garbled status),
    that's a 'couldn't classify' signal, not a confident judgment either
    way — must not silently resolve to 100 (masking a real problem) or to
    0 (unfairly tanking an otherwise-fine brochure)."""
    score = factual_consistency_score_from_status("not_a_real_status")
    assert 0 < score < 100
    assert score != 100.0


def test_missing_status_also_falls_back_safely():
    assert 0 < factual_consistency_score_from_status("") < 100
    assert 0 < factual_consistency_score_from_status(None) < 100


def test_status_matching_is_not_case_or_whitespace_sensitive():
    assert factual_consistency_score_from_status("Consistent") == 100.0
    assert factual_consistency_score_from_status("  consistent  ") == 100.0
    assert factual_consistency_score_from_status("CONTRADICTION") == 20.0
