"""Node 2: the ledger takes the form's own fallback without asking a model
when it should not, and falls back on its own when the model fails."""
from __future__ import annotations

import pytest

from src.agents.brochure import property_intelligence
from src.agents.video import fact_ledger
from src.core.exceptions import ModelProviderError
from tests.video.helpers import make_video_property
from src.agents.video import brief as brief_agent


@pytest.fixture
def prop(tmp_path):
    return brief_agent.run(make_video_property(tmp_path)).property_input


@pytest.fixture
def refuse_models(monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a model was asked")

    monkeypatch.setattr("src.agents.brochure.property_intelligence.call_with_fallback", refuse)
    return calls


def test_an_offline_run_builds_the_ledger_from_the_form_and_asks_no_model(prop, refuse_models, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "a-key-that-must-not-be-used", raising=False)

    knowledge, notes = fact_ledger.run(prop, "", offline=True)

    assert refuse_models == []
    assert knowledge == property_intelligence.ledger_from_form(prop)
    assert knowledge.facts["name"] == "Aurelia Heights" and knowledge.rera_number == "P52100012345"
    assert notes == ["Fact ledger built from the form (offline run)."]


def test_with_no_model_key_the_form_ledger_is_taken_without_waiting_for_retries(prop, refuse_models):
    knowledge, notes = fact_ledger.run(prop, "", offline=False)
    assert refuse_models == []
    assert "no model key is set" in notes[0] and knowledge.facts["name"] == "Aurelia Heights"


def test_documents_the_form_ledger_cannot_read_are_said_not_to_be_used(prop, refuse_models):
    _, notes = fact_ledger.run(prop, "=== brochure.pdf ===\nSome text", offline=True)
    assert any("documents were not read" in n for n in notes)


def test_online_the_model_is_asked_and_its_ledger_is_used(prop, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "k", raising=False)
    asked = []

    def answer(role, fn_name, prompt):
        asked.append(role)
        return {"facts": {"name": "Aurelia Heights", "floors": 31}, "selling_points": ["Metro nearby"],
                "specifications": [], "legal_notes": [], "rera_number": ""}

    monkeypatch.setattr("src.agents.brochure.property_intelligence.call_with_fallback", answer)

    knowledge, notes = fact_ledger.run(prop, "", offline=False)

    assert len(asked) == 1
    assert knowledge.facts["floors"] == "31" and knowledge.rera_number == "P52100012345"
    assert notes == ["Fact ledger read from the form and documents."]


def test_online_a_failing_model_still_ends_in_the_form_ledger(prop, monkeypatch):
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "k", raising=False)
    monkeypatch.setattr("src.agents.base.time.sleep", lambda s: None)

    def fail(*args, **kwargs):
        raise ModelProviderError("quota")

    monkeypatch.setattr("src.agents.brochure.property_intelligence.call_with_fallback", fail)

    knowledge, _ = fact_ledger.run(prop, "", offline=False)

    assert knowledge == property_intelligence.ledger_from_form(prop)
