"""Plan, write, render, measure, repair — one brochure.

    1. plan the whole book from the material       planner.py
    2. write every page from the facts              copywriter.py
    3. render it in Chromium, measure what a        brochure_audit.js, qa.py
       reader would see, and repair by whole
       sentences — up to REPAIR_ROUNDS times
    4. render the delivered files: the reading PDF, the print-ready PDF
       (3mm bleed, crop marks) and one image per page

Used by the LangGraph workflow (src/workflows/brochure.py) and by the
offline bench (scripts/bench_brochure.py), so the two cannot disagree.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.agents.brochure import copywriter, planner, qa, strategy
from src.agents.brochure.book import Book
from src.agents.brochure.catalog import TemplateSpec
from src.agents.brochure.palettes import Palette
from src.rendering.brochure_renderer import render_book
from src.rendering.renderer import render_pdf_and_screenshot

logger = logging.getLogger(__name__)

REPAIR_ROUNDS = 4


@dataclass
class Production:
    book: Book
    html_path: Path
    pdf_path: Path
    screenshot_path: Path
    report: qa.Report
    repairs: list[str] = field(default_factory=list)
    rounds: int = 0


def prepare_assets(images: list, analyses: dict, out_dir: Path, template: str) -> tuple[list, list[str]]:
    """Normalise every image once, before it is placed: drawings flattened
    and trimmed of blank canvas (so a plan fills its card), the category's
    grade on photographs, oversized files reduced to print resolution.
    Logos keep their transparency — they are trimmed only."""
    from src.data import grading
    from src.data.asset_prep import burned_in_text_warning, normalise_image, trim_logo

    grade = grading.for_template(template)
    out, notes = [], []
    for image in images:
        if image.role == "logo" or "logo" in image.filename.lower():
            path = trim_logo(Path(image.path), out_dir, image.image_id)
            out.append(image.model_copy(update={"path": str(path)}) if str(path) != image.path else image)
            continue
        analysis = analyses.get(image.image_id)
        category = (getattr(analysis.category, "value", None) or str(analysis.category)) if analysis else ""
        if image.role in ("floorplan", "masterplan", "map"):
            category = "floorplan" if image.role != "map" else "map"
        path, issues = normalise_image(Path(image.path), out_dir, image.image_id, category, grade=grade)
        notes.extend(i.message for i in issues if i.kind == "burned_in_text")
        if analysis is not None:
            warning = burned_in_text_warning(image.image_id, analysis.visible_text, category)
            if warning:
                notes.append(warning.message)
        out.append(image.model_copy(update={"path": str(path)}) if str(path) != image.path else image)
    return out, notes


def _audit(screenshot_path: Path) -> dict:
    path = Path(str(screenshot_path) + ".layout.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def produce(prop, images: list, analyses: dict, spec: TemplateSpec, palette: Palette,
            out_dir: Path, *, use_model: bool = True, rounds: int = REPAIR_ROUNDS,
            book: Optional[Book] = None) -> Production:
    positioning = strategy.for_category(spec.category)
    images_by_id = {i.image_id: i for i in images}
    if book is None:
        book = planner.plan(prop, images, analyses, spec, positioning)
        book.palette = palette.name
        copywriter.write(book, prop, positioning, use_model=use_model)

    html_path = out_dir / "index.html"
    pdf_path = out_dir / "brochure.pdf"
    shot = out_dir / "screenshot.png"
    repairs: list[str] = []
    used_rounds = 0

    def measure(final: bool = False) -> qa.Report:
        render_book(book, prop, spec, palette, images_by_id, html_path)
        render_pdf_and_screenshot(html_path, pdf_path, shot, capture_pages=final)
        return qa.evaluate(_audit(shot), book)

    report = measure()
    blocked: dict[int, set] = {}
    labels = {"sentence": "one sentence fewer", "point": "one point fewer", "lead": "standfirst dropped",
              "figure": "one figure fewer", "place": "one place fewer", "amenity": "one amenity fewer",
              "card_text": "card details beside key figures dropped",
              "card_text_any": "a card detail dropped", "dev": "developer line dropped",
              "note": "footnote dropped", "kicker": "kicker dropped", "grow": "one sentence more",
              "expand": "photograph enlarged into unused space"}
    for round_no in range(rounds + 2):
        before = {p.number: p for p in report.pages}
        snapshot = {p.number: p.model_copy(deep=True) for p in book.pages}
        changed = qa.repair(book, report, blocked)
        if not changed:
            break
        used_rounds = round_no + 1
        report = measure()
        after = {p.number: p for p in report.pages}
        for number, action in changed.items():
            old, new = before.get(number), after.get(number)
            if old is None or new is None:
                continue
            worse = len(new.defects) > len(old.defects) or \
                    (action in ("grow", "expand") and new.overflow_mm > qa.OVERFLOW_TOLERANCE_MM) or \
                    (action != "grow" and new.overflow_mm > old.overflow_mm - 0.4
                     and old.overflow_mm > qa.OVERFLOW_TOLERANCE_MM)
            index = next(i for i, p in enumerate(book.pages) if p.number == number)
            if worse:
                # it did not help: put the page back as it was, and do not try it again
                book.pages[index] = snapshot[number]
                blocked.setdefault(number, set()).add(action)
                if action == "grow":
                    blocked[number].add("sentence")
            else:
                repairs.append(f"round {round_no + 1}: page {number}: {labels.get(action, action)}")
    report = measure(final=True)
    qa.check_pdf(pdf_path, book, report)
    return Production(book=book, html_path=html_path, pdf_path=pdf_path, screenshot_path=shot,
                      report=report, repairs=repairs, rounds=used_rounds)
