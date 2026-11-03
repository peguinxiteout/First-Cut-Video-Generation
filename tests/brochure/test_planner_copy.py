"""The book is planned from the material (pages, pictures, slots) and written
from the facts — no generic pages, no picture twice, no invented number,
no sentence twice, a voice per category."""
from __future__ import annotations

import re

import pytest

from src.agents.brochure import catalog, copywriter, planner, strategy, vision_offline
from src.data.ingestion import discover_images, load_property_input
from tests.brochure.helpers import make_property

CATEGORY = {"affordable": "Affordable", "mid": "Mid-Tier", "luxury": "Luxury"}


def _book(tmp_path, template_id, **extra):
    spec = catalog.resolve(template_id)
    folder = make_property(tmp_path / template_id, category=CATEGORY[spec.category],
                           orientation=spec.orientation.title(), **extra)
    prop = load_property_input(folder)
    images = discover_images(folder)
    book = planner.plan(prop, images, vision_offline.run(images), spec)
    copywriter.write(book, prop, use_model=False)
    return prop, images, spec, book


@pytest.fixture(params=list(catalog.TEMPLATES))
def template_id(request):
    return request.param


def test_every_template_plans_a_complete_book(tmp_path, template_id):
    prop, images, spec, book = _book(tmp_path, template_id)
    kinds = [p.kind for p in book.pages]
    assert kinds[0] == "cover" and kinds[-1] == "closing"
    assert "contents" not in kinds and "booking" not in kinds
    assert len(book.pages) <= strategy.for_category(spec.category).max_pages
    # every slot lies on the sheet
    for page in book.pages:
        for p in page.images:
            assert -0.01 <= p.x and p.x + p.w <= spec.sheet.w + 0.01
            assert -0.01 <= p.y and p.y + p.h <= spec.sheet.h + 0.01
        for b in page.boxes.values():
            assert b.x >= 0 and b.x + b.w <= spec.sheet.w + 0.01 and b.y + b.h <= spec.sheet.h + 0.01


def test_no_photograph_is_printed_twice(tmp_path, template_id):
    _prop, _images, _spec, book = _book(tmp_path, template_id)
    placed = [p.image_id for page in book.pages for p in page.images]
    assert len(placed) == len(set(placed))


def test_plans_are_shown_whole_and_photographs_crop_purposefully(tmp_path):
    _prop, _images, _spec, book = _book(tmp_path, "mid_landscape", plans=2)
    for page in book.pages:
        for p in page.images:
            if p.kind == "plan":
                assert p.crop == 0
            else:
                assert p.crop < 0.55, (page.kind, p.crop)


def test_the_masterplan_is_given_the_page(tmp_path):
    from tests.brochure.helpers import plan_bytes
    from src.data import property_form
    from tests.brochure.helpers import PHOTOS, form, photo_bytes

    uploads = {"photo": [(n, photo_bytes(s, c, i)) for i, (n, s, c) in enumerate(PHOTOS)],
               "masterplan": [("site.png", plan_bytes((1600, 1100)))]}
    folder = property_form.write_property(form("Luxury", "Landscape"), uploads, tmp_path)
    prop = load_property_input(folder)
    images = discover_images(folder)
    spec = catalog.resolve("luxury_landscape")
    book = planner.plan(prop, images, vision_offline.run(images), spec)
    page = next(p for p in book.pages if p.kind == "masterplan")
    plan = page.images[0]
    frame = spec.frame()
    assert plan.kind == "plan" and plan.w >= frame.w * 0.7          # the drawing takes the page
    biggest_photo = max(p.w * p.h for pg in book.pages if pg.kind not in ("cover", "closing")
                        for p in pg.images if p.kind == "image")
    assert plan.w * plan.h >= biggest_photo * 0.6


def test_luxury_never_prints_a_price_and_affordable_leads_with_it(tmp_path):
    _p, _i, _s, lux = _book(tmp_path, "luxury_portrait")
    words = " ".join(_all_text(p) for p in lux.pages)
    assert "₹" not in words and "Lakh" not in words
    assert not any(u.get("price") for p in lux.pages for u in p.data.get("units", []))
    _p, _i, _s, aff = _book(tmp_path, "affordable_landscape")
    assert aff.pages[0].data.get("price")


def test_no_sentence_or_title_appears_twice(tmp_path, template_id):
    _prop, _images, _spec, book = _book(tmp_path, template_id)
    seen = {}
    for page in book.pages:
        w = page.words
        for text in [w.title, w.lead] + [s for b in w.body for s in re.split(r"(?<=[.!?])\s+", b)]:
            key = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
            if len(key) < 12:
                continue
            assert key not in seen, f"'{text}' on pages {seen.get(key)} and {page.number}"
            seen[key] = page.number


def test_no_number_the_property_did_not_give(tmp_path, template_id):
    prop, _images, _spec, book = _book(tmp_path, template_id)
    source = prop.model_dump_json().replace(",", "")
    for page in book.pages:
        w = page.words
        text = " ".join([w.title, w.lead] + w.body + [c.title + " " + c.text for c in w.cards])
        for number in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
            assert number.replace(",", "") in source, (page.kind, number, text)


def test_raw_configurations_never_reach_the_page(tmp_path):
    from src.data import property_form

    rows = [property_form.ConfigRow(configuration="1 Bed + 1 Toilet", carpet_area="434 sq.ft"),
            property_form.ConfigRow(configuration="3 Bed + 3 Toilet + Study", carpet_area="1270 sq.ft")]
    _p, _i, _s, book = _book(tmp_path, "mid_landscape", configurations=rows)
    text = " ".join(_all_text(p) for p in book.pages)
    assert "Toilet" not in text and "Bed +" not in text
    units = next(p for p in book.pages if p.kind == "residences").data["units"]
    assert [u["label"] for u in units] == ["1 BHK", "3 BHK + Study"]
    assert units[1]["area"] == "1,270 sq ft"


def test_the_three_voices_differ_on_the_same_property(tmp_path):
    leads = {}
    for cat in ("affordable_landscape", "mid_landscape", "luxury_landscape"):
        _p, _i, _s, book = _book(tmp_path, cat)
        overview = next(p for p in book.pages if p.kind == "overview")
        leads[cat] = (overview.words.title, overview.words.lead)
    assert len({t for t, _ in leads.values()}) == 3
    assert len({lead for _, lead in leads.values()}) == 3


def test_two_projects_in_one_category_do_not_read_alike(tmp_path):
    _p, _i, _s, a = _book(tmp_path / "a", "mid_landscape")
    _p, _i, _s, b = _book(tmp_path / "b", "mid_landscape", name="Juniper Grove", city="Hyderabad",
                          locality="Kokapet", target_audience="young professionals and growing families",
                          tagline="Greener by nature",
                          amenities="Tennis court\nYoga deck\nCo-working lounge\nPet park",
                          nearby="Financial District | 4 km | work\nKokapet Lake | 1 km",
                          usps="Pre-certified IGBC Platinum design\nEvery home faces the lake\n"
                               "Private lift lobby for every two homes")
    text_a = {s for p in a.pages for s in [p.words.lead] + p.words.body if s}
    text_b = {s for p in b.pages for s in [p.words.lead] + p.words.body if s}
    assert len(text_a & text_b) <= len(text_a) // 3


def test_each_selling_point_is_on_exactly_one_page(tmp_path, template_id):
    prop, _images, _spec, book = _book(tmp_path, template_id)
    for usp in prop.usps:
        head = usp.split()[0].lower()
        pages = [p.number for p in book.pages
                 if any(head in (c.title + " " + c.text).lower() for c in p.words.cards)]
        assert len(pages) <= 1, (usp, pages)


def _all_text(page) -> str:
    w = page.words
    parts = [w.kicker, w.title, w.lead, w.note] + w.body + [c.title + " " + c.text + " " + c.value for c in w.cards]
    for key in ("units", "figures", "summary"):
        for row in page.data.get(key, []) or []:
            parts.extend(str(v) for v in row.values())
    return " ".join(x for x in parts if x)
