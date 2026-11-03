"""The model writer: one call for the book, every field held to the guard."""
from __future__ import annotations

from src.agents.brochure import catalog, copywriter, planner, vision_offline
from src.data.ingestion import discover_images, load_property_input
from tests.brochure.helpers import make_property


def _book(tmp_path):
    folder = make_property(tmp_path, category="Mid-Tier")
    prop = load_property_input(folder)
    images = discover_images(folder)
    spec = catalog.resolve("mid_landscape")
    return prop, planner.plan(prop, images, vision_offline.run(images), spec)


def test_model_fields_are_taken_only_when_they_pass_the_guard(tmp_path, monkeypatch):
    prop, book = _book(tmp_path)
    overview = next(p for p in book.pages if p.kind == "overview")

    def fake(role, method, prompt):
        assert "Aurelia Heights" in prompt and method == "generate_json"
        return {"pages": [{
            "page": overview.number,
            "title": "Baner, beautifully balanced",
            "lead": "A home 400 metres from everything.",                    # invented number: rejected
            "sentences": ["Mornings start by the pool and end in the yoga lawn.",
                          "Two minutes from the Baner metro station, the city is close.",
                          "This image shows a tower."],                       # describes the picture: rejected
        }]}

    monkeypatch.setattr("src.agents.brochure.copywriter.call_with_fallback", fake)
    copywriter.write(book, prop, use_model=True)
    w = overview.words
    assert w.title == "Baner, beautifully balanced" and w.source == "model"
    assert "400" not in w.lead                       # the offline standfirst was kept
    body = " ".join(w.body)
    assert "This image shows" not in body
    assert "Mornings start by the pool" in " ".join(overview.pool)


def test_without_a_model_every_page_is_still_written(tmp_path, monkeypatch):
    prop, book = _book(tmp_path)

    def refuse(*_a, **_k):
        raise RuntimeError("no model")

    monkeypatch.setattr("src.agents.brochure.copywriter.call_with_fallback", refuse)
    copywriter.write(book, prop, use_model=True)
    for page in book.pages:
        assert page.words.title or page.kind == "plans", page.kind
        assert page.words.source == "offline"
