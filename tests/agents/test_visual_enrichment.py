"""Supporting AI visuals: only where a gap exists, flagged as generated,
never page furniture, never labelled an "artistic impression", and never
on the cover, the back, the contents or the overview."""
from __future__ import annotations

from src.agents.brochure import catalog, planner, strategy, visual_enrichment, vision_offline
from src.data.ingestion import discover_images, load_property_input
from src.data.schemas import PropertyKnowledge
from src.models.image_gen import ImageService
from src.models.image_gen.mock import MockImageProvider
from tests.brochure.helpers import make_property

KNOWLEDGE = PropertyKnowledge(selling_points=["Sixty percent open green space"])


def _briefs(kinds):
    subjects = {"design_element": "close crop of warm brick coursing in raking light",
                "lifestyle": "a couple on a terrace at golden hour, seen from behind",
                "neighbourhood": "a tree-lined boulevard at dusk",
                "amenity_concept": "a quiet yoga lawn at sunrise"}
    return lambda role, fn, prompt: {"images": [
        {"kind": k, "purpose": "gap", "subject": subjects[k], "aspect": "landscape", "caption": ""}
        for k in kinds]}


def _run(tmp_path, monkeypatch, kinds):
    monkeypatch.setattr("src.agents.image.gap_analysis.call_with_fallback", _briefs(kinds))
    folder = make_property(tmp_path / "props", photos=4)
    prop = load_property_input(folder)
    images = discover_images(folder)
    analyses = vision_offline.run(images)
    service = ImageService(MockImageProvider(), tmp_path / "cache")
    return prop, visual_enrichment.run(prop, KNOWLEDGE, images, analyses, service, tmp_path / "ai")


def test_generated_visuals_are_flagged_and_never_furniture(tmp_path, monkeypatch):
    prop, (images, analyses, report) = _run(tmp_path, monkeypatch,
                                            ["design_element", "lifestyle", "neighbourhood"])
    ai = [i for i in images if i.is_ai_generated]
    assert len(ai) == 2                                    # the design element is not made
    for image in ai:
        note = analyses[image.image_id].notes.lower()
        assert "artistic impression" not in note and "not a depiction of the property" in note
        assert analyses[image.image_id].hero_score == 0


def test_generated_visuals_never_take_the_cover_or_the_overview(tmp_path, monkeypatch):
    prop, (images, analyses, _report) = _run(tmp_path, monkeypatch,
                                             ["lifestyle", "neighbourhood", "amenity_concept"])
    spec = catalog.resolve("mid_portrait")
    book = planner.plan(prop, images, analyses, spec, strategy.for_category("mid"))
    ai = {i.image_id for i in images if i.is_ai_generated}
    for page in book.pages:
        if page.kind in ("cover", "closing", "overview"):
            assert not ai & {p.image_id for p in page.images}, page.kind
