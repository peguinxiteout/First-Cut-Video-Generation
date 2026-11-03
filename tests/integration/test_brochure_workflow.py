"""The whole brochure workflow, as the Streamlit app runs it, with no API
key: every model step falls back and a complete, measured brochure still
comes out, with its record, its evaluation and what each agent did."""
from __future__ import annotations

import pytest

from src.core.types import QualityGateResult
from src.data import property_form, repository
from src.workflows import brochure as workflow
from tests.brochure.helpers import make_property
from tests.brochure.test_render_and_audit import _chromium_available

pytestmark = pytest.mark.skipif(not _chromium_available(), reason="Playwright Chromium not installed")


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    from src.core.exceptions import ModelProviderError

    def refuse(*_args, **_kwargs):
        raise ModelProviderError("offline test")

    for target in ("src.agents.brochure.property_intelligence.call_with_fallback",
                   "src.agents.brochure.vision.call_with_fallback",
                   "src.agents.brochure.creative_director.call_with_fallback",
                   "src.agents.brochure.copywriter.call_with_fallback",
                   "src.agents.brochure.critic.call_with_fallback",
                   "src.agents.image.gap_analysis.call_with_fallback"):
        monkeypatch.setattr(target, refuse, raising=False)


def test_the_form_to_pdf_path(tmp_path):
    folder = make_property(tmp_path / "properties", category="Luxury", orientation="Portrait")
    template = property_form.template_for(property_form.BrochureForm(category="Luxury", orientation="Portrait"))
    record = workflow.run_brochure_workflow(folder, template_name=template)

    assert record.status == "completed", record.error
    assert record.template == "Luxury - Portrait"
    assert record.pdf_path and record.page_count >= 6
    assert record.workflow_evaluation.all_succeeded
    assert [s.node for s in record.workflow_evaluation.steps][:2] == ["property_intelligence", "vision"]
    assert record.evaluation is not None and record.evaluation.judge == "measured"
    assert record.evaluation.gate_result in tuple(QualityGateResult)
    assert set(record.agent_reports) >= {"art_director", "copywriter", "render_qa", "critic"}
    # nothing technical leaks into what the person reads about the agents
    lines = " ".join(line for lines in record.agent_reports.values() for line in lines)
    assert "step_up" not in lines and "round 1" not in lines

    stored = repository.get_generation(record.generation_id)
    assert stored.template == record.template and stored.page_count == record.page_count
