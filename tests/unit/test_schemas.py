from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.types import ImageCategory, QualityGateResult
from src.data.schemas import (
    EvaluationResult,
    ImageAnalysis,
    PropertyInput,
    PropertyLocation,
)


def test_property_input_requires_location():
    with pytest.raises(ValidationError):
        PropertyInput(property_id="P1", name="Test")  # missing location


def test_property_input_minimal_valid(sample_property_input):
    assert sample_property_input.property_id == "PROP-TEST"
    assert sample_property_input.location.city == "Ahmedabad"


def test_image_analysis_score_bounds():
    with pytest.raises(ValidationError):
        ImageAnalysis(
            image_id="IMG001",
            category=ImageCategory.EXTERIOR,
            quality=15,  # out of 0-10 range
            composition=5,
            hero_score=5,
        )


def test_image_analysis_valid():
    analysis = ImageAnalysis(
        image_id="IMG001",
        category=ImageCategory.FLOORPLAN,
        quality=8.5,
        composition=7.0,
        hero_score=2.0,
    )
    assert analysis.category == ImageCategory.FLOORPLAN


def test_evaluation_result_defaults_to_fail_gate():
    evaluation = EvaluationResult()
    assert evaluation.gate_result == QualityGateResult.FAIL
    assert evaluation.quality_score == 0
