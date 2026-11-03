"""End to end in Chromium: the brochure rendered, measured and repaired,
and the audit shown to catch each defect it exists to catch.

Needs Playwright's Chromium (skipped if it is not installed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.brochure import catalog, palettes, production, qa, vision_offline
from src.data.ingestion import discover_images, load_property_input
from tests.brochure.helpers import make_property

AUDIT_JS = Path(__file__).resolve().parents[2] / "src" / "rendering" / "brochure_audit.js"
CATEGORY = {"affordable": "Affordable", "mid": "Mid-Tier", "luxury": "Luxury"}
ORIENTATION = {"landscape": "Landscape", "portrait": "Portrait", "digital": "Digital 16:9"}


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _chromium_available(), reason="Playwright Chromium not installed")


def _produce(tmp_path, template_id, rounds=production.REPAIR_ROUNDS):
    spec = catalog.resolve(template_id)
    folder = make_property(tmp_path / "props", category=CATEGORY[spec.category],
                           orientation=ORIENTATION[spec.orientation])
    prop = load_property_input(folder)
    images = discover_images(folder)
    analyses = vision_offline.run(images)
    images, _notes = production.prepare_assets(images, analyses, tmp_path / "norm", spec.id)
    return production.produce(prop, images, analyses, spec, palettes.for_category(spec.category),
                              tmp_path / "out", use_model=False, rounds=rounds)


@pytest.mark.parametrize("template_id", ["affordable_portrait", "mid_landscape", "luxury_digital"])
def test_a_brochure_renders_with_no_defect(tmp_path, template_id):
    result = _produce(tmp_path, template_id)
    assert result.pdf_path.exists() and result.pdf_path.stat().st_size > 5000
    assert result.pdf_path.with_name("brochure-print.pdf").exists()          # bleed + crop marks
    assert len(result.report.pages) == len(result.book.pages)
    assert result.report.defects == []
    shots = sorted((tmp_path / "out" / "pages").glob("page_*.png"))
    assert len(shots) == len(result.book.pages)
    if template_id.endswith("digital"):
        from PIL import Image

        with Image.open(shots[0]) as im:
            assert im.size == (1920, 1080)


def test_the_pdf_is_a4(tmp_path):
    fitz = pytest.importorskip("fitz")
    result = _produce(tmp_path, "affordable_landscape", rounds=1)
    doc = fitz.open(str(result.pdf_path))
    w_mm, h_mm = doc[0].rect.width * 25.4 / 72, doc[0].rect.height * 25.4 / 72
    assert abs(w_mm - 297) < 1.5 and abs(h_mm - 210) < 1.5
    # the text is real text, and the rupee sign survives into the PDF
    text = "".join(page.get_text() for page in doc)
    assert "₹" in text and "BHK" in text


def _audit_after(html_path: Path, script: str) -> dict:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        page.goto(html_path.resolve().as_uri())
        page.wait_for_load_state("networkidle")
        page.evaluate("async () => { await document.fonts.ready; }")
        page.evaluate(script)
        page.add_script_tag(content=AUDIT_JS.read_text(encoding="utf-8"))
        report = page.evaluate("auditBrochure(document, window)")
        browser.close()
    return report


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("audit")
    return _produce(tmp, "mid_landscape", rounds=0).html_path


def test_the_audit_passes_the_rendered_book(rendered):
    report = qa.evaluate(_audit_after(rendered, "0"))
    assert not [d for d in report.defects if "breaks" in d or "did not load" in d or "6pt" in d]
    assert all(report_ok for report_ok in _audit_after(rendered, "0")["fonts"].values())


def test_the_audit_catches_copy_that_overflows(rendered):
    report = qa.evaluate(_audit_after(rendered, """
        const box = document.querySelector('.k-overview .box.text .col-body') || document.querySelector('.box.text');
        for (let i = 0; i < 40; i++) { const p = document.createElement('p'); p.textContent = 'An extra line of copy ' + i; box.appendChild(p); }
    """))
    assert any("runs past its box" in d for d in report.defects)


def test_the_audit_catches_a_broken_unit(rendered):
    report = qa.evaluate(_audit_after(rendered, """
        const k = [...document.querySelectorAll('.keep')].find(e => e.textContent.trim().includes(' '));
        k.style.whiteSpace = 'normal'; k.style.display = 'inline'; k.parentElement.style.display = 'block'; k.parentElement.style.width = '6mm';
    """))
    assert any("breaks across lines" in d for d in report.defects)


def test_the_audit_catches_a_missing_image(rendered):
    report = qa.evaluate(_audit_after(rendered, """async () => {
        const img = document.querySelector('img[data-img]');
        img.src = 'file:///nowhere/missing.jpg';
        await new Promise(r => setTimeout(r, 300));
    }"""))
    assert any("did not load" in d for d in report.defects)


def test_the_qa_report_reads_the_audit():
    audit = {"pages": [{"page": 2, "kind": "feature", "whitespace": 0.9,
                        "boxes": [{"name": "text", "overflow_mm": 4.2, "fill": 1.1}],
                        "broken": ["1 BHK · Type A"], "images": [], "small_text": [], "outside": []}],
             "texts": [{"page": 2, "text": "A home for every season"},
                       {"page": 3, "text": "A home for every season"}],
             "fonts": {"Inter": True, "Jost": False}}
    report = qa.evaluate(audit)
    assert not report.passed
    assert any("runs past its box" in d for d in report.defects)
    assert any("breaks across lines" in d for d in report.defects)
    assert any("same text appears" in d for d in report.defects)
    assert any("typeface did not load" in d and "Jost" in d for d in report.defects)
    assert qa.unresolved(report)
