"""The brochure's measurable qualities, from the rendered pages.

The requirements that can be checked without anybody's opinion:

    page boundaries   no copy past its box or off the page
    typesetting       no unit, price, area or phone number broken across
                      lines; nothing below 6pt; every typeface loaded (so
                      the rupee sign never falls back to a missing glyph)
    image integrity   every image loaded
    repetition        no title or paragraph printed twice
    white space       each page within its category's range (Luxury is
                      meant to be open; Affordable compact)

They are scored here so that a brochure is evaluated even when no model
judge is available, and so that a model judge cannot rate a brochure well
that measurably breaks a rule.
"""
from __future__ import annotations

from src.agents.brochure import qa
from src.agents.brochure.catalog import DESIGNS

CHECKS = (
    ("boundaries", ("runs past its box", "runs off the page", "set over a photograph")),
    ("typesetting", ("breaks across lines", "below 6pt", "typeface did not load")),
    ("image_integrity", ("image did not load",)),
    ("repetition", ("same text appears",)),
)

LABELS = {
    "boundaries": "Copy within its boxes and the page, clear of the photographs",
    "typesetting": "No broken units or prices; fonts and ₹ glyph loaded",
    "image_integrity": "Every image loaded",
    "repetition": "No repeated text",
    "whitespace": "Open space within the category's range",
    "resolution": "Photographs at print resolution",
}


def _band(report: qa.Report, category: str) -> tuple[float, float]:
    design = DESIGNS.get(category)
    lo, hi = design.whitespace if design else (0.2, 0.4)
    return lo - 0.12, hi + 0.15


def measure(report: qa.Report, category: str = "mid") -> dict:
    checks: dict[str, str] = {}
    for name, needles in CHECKS:
        hits = [d for d in report.defects if any(n in d for n in needles)]
        checks[name] = "pass" if not hits else f"{len(hits)} problem(s): " + "; ".join(hits[:3])
    body = [p for p in report.pages if p.kind not in ("cover", "closing")]
    lo, hi = _band(report, category)
    outside = [p for p in body if not (lo <= p.whitespace <= hi)]
    mean_ws = sum(p.whitespace for p in body) / len(body) if body else 0.0
    checks["whitespace"] = "pass" if not outside else (
        f"{len(outside)} of {len(body)} pages outside {lo:.0%}–{hi:.0%}")
    soft = [a for a in report.advisories if "dpi" in a]
    checks["resolution"] = "pass" if not soft else f"{len(soft)} photograph(s) below 150 dpi"

    layout = max(0.0, 100.0 - 12.0 * len(report.defects))
    image_integrity = 100.0 if checks["image_integrity"] == "pass" else max(
        0.0, 100.0 - 25.0 * sum(1 for d in report.defects if "image" in d))
    balance = max(0.0, 100.0 - 6.0 * len(outside))
    resolution = max(0.0, 100.0 - 5.0 * len(soft))
    return {
        "checks": checks,
        "layout_integrity": round(layout, 1),
        "image_integrity": round(image_integrity, 1),
        "whitespace_mean": round(mean_ws, 3),
        "pages_over_whitespace": len(outside),
        "visual_proxy": round(0.35 * layout + 0.25 * balance + 0.2 * image_integrity + 0.2 * resolution, 1),
    }


def plain_issues(report: qa.Report) -> list[str]:
    """What a person reviewing the brochure should know, in plain words."""
    return list(report.defects) + [a for a in report.advisories if "dpi" in a or "open space" in a]
