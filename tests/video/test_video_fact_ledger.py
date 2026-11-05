"""Node 2: the ledger takes the form's own fallback without asking a model
when it should not, and falls back on its own when the model fails."""
from __future__ import annotations

import pytest

from src.agents.brochure import property_intelligence
from src.agents.video import fact_ledger
from src.core.config import settings
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


# -- the cache -----------------------------------------------------------------------------------------------------

ANSWER = {"facts": {"name": "Aurelia Heights", "floors": 31}, "selling_points": ["Metro nearby"],
          "specifications": [], "legal_notes": [], "rera_number": ""}


@pytest.fixture
def asked(monkeypatch):
    """A model that answers (and counts how many times it was asked)."""
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "k", raising=False)
    calls = []

    def answer(role, fn_name, prompt):
        calls.append(prompt)
        return {**ANSWER, "selling_points": [f"Metro nearby ({len(calls)})"]}

    monkeypatch.setattr("src.agents.brochure.property_intelligence.call_with_fallback", answer)
    return calls


def test_the_second_run_of_a_property_reads_its_ledger_from_the_cache_and_says_so(prop, asked, tmp_path):
    first, first_notes = fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    again, notes = fact_ledger.run(prop, "", offline=False, directory=tmp_path)

    assert len(asked) == 1 and again == first                              # the model was asked once
    assert first_notes == ["Fact ledger read from the form and documents."]
    assert notes[0].startswith(fact_ledger.CACHE_NOTE) and "--refresh-ledger" in notes[0]
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_the_cache_is_under_the_data_folder_by_default(prop, asked):
    fact_ledger.run(prop, "", offline=False)
    assert len(list(fact_ledger.cache_dir().glob("*.json"))) == 1
    assert fact_ledger.cache_dir().parent == settings.data_dir_resolved and fact_ledger.cache_dir().name == "ledger_cache"


def test_a_change_to_the_property_the_documents_the_model_or_the_prompt_is_asked_again(prop, asked, tmp_path, monkeypatch):
    fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    fact_ledger.run(prop.model_copy(update={"price": "Rs 9 crore"}), "", offline=False, directory=tmp_path)
    assert len(asked) == 2                                                  # the property's data
    fact_ledger.run(prop, "=== brochure.pdf ===\nSome text", offline=False, directory=tmp_path)
    assert len(asked) == 3                                                  # its documents
    monkeypatch.setattr("src.core.config.settings.gemini_text_model", "another-model", raising=False)
    fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    assert len(asked) == 4                                                  # the model
    monkeypatch.setattr(fact_ledger, "CACHE_VERSION", 2)
    fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    assert len(asked) == 5                                                  # the prompt's version
    fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    assert len(asked) == 5 and len(list(tmp_path.glob("*.json"))) == 5      # and each of them is kept


def test_the_key_is_the_property_data_the_documents_the_model_and_the_version(prop):
    base = fact_ledger.cache_key(prop, "docs", "gemini:m", "v1")
    assert base == fact_ledger.cache_key(prop.model_copy(), "docs", "gemini:m", "v1")
    others = {fact_ledger.cache_key(prop.model_copy(update={"name": "Other"}), "docs", "gemini:m", "v1"),
              fact_ledger.cache_key(prop, "other docs", "gemini:m", "v1"), fact_ledger.cache_key(prop, "docs", "gemini:n", "v1"),
              fact_ledger.cache_key(prop, "docs", "gemini:m", "v2")}
    assert base not in others and len(others) == 4


def test_refresh_asks_the_model_again_and_keeps_the_new_answer(prop, asked, tmp_path):
    first, _ = fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    fresh, notes = fact_ledger.run(prop, "", offline=False, refresh=True, directory=tmp_path)
    assert len(asked) == 2 and fresh != first
    assert notes == [f"{fact_ledger.REFRESH_NOTE}: the model was asked again and its answer kept."]
    later, later_notes = fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    assert len(asked) == 2 and later == fresh and later_notes[0].startswith(fact_ledger.CACHE_NOTE)


def test_a_ledger_that_is_only_the_form_fallback_is_not_kept(prop, monkeypatch, tmp_path):
    monkeypatch.setattr("src.core.config.settings.gemini_api_key", "k", raising=False)
    monkeypatch.setattr("src.agents.base.time.sleep", lambda s: None)
    monkeypatch.setattr("src.agents.brochure.property_intelligence.call_with_fallback",
                        lambda *a, **k: (_ for _ in ()).throw(ModelProviderError("quota")))
    knowledge, _ = fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    assert knowledge == property_intelligence.ledger_from_form(prop) and list(tmp_path.glob("*.json")) == []
    _, notes = fact_ledger.run(prop, "", offline=False, refresh=True, directory=tmp_path)
    assert "gave nothing the form does not" in notes[0]


def test_an_offline_run_neither_reads_nor_writes_the_cache(prop, asked, tmp_path):
    fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    _, notes = fact_ledger.run(prop, "", offline=True, directory=tmp_path)
    assert notes == ["Fact ledger built from the form (offline run)."] and len(asked) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_an_unreadable_or_wrong_cache_entry_is_asked_again_not_trusted(prop, asked, tmp_path):
    fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    (entry,) = tmp_path.glob("*.json")
    entry.write_text("{not json", encoding="utf-8")
    fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    assert len(asked) == 2
    entry.write_text('{"knowledge": {"facts": 7}}', encoding="utf-8")
    fact_ledger.run(prop, "", offline=False, directory=tmp_path)
    assert len(asked) == 3


def test_a_cache_that_cannot_be_written_does_not_fail_the_ledger(prop, asked, tmp_path):
    blocked = tmp_path / "file_not_folder"
    blocked.write_text("x", encoding="utf-8")
    knowledge, notes = fact_ledger.run(prop, "", offline=False, directory=blocked / "sub")
    assert knowledge.facts["floors"] == "31" and notes == ["Fact ledger read from the form and documents."]
