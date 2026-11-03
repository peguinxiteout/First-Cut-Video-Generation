"""Render a brochure end to end OFFLINE, in any template.

    python scripts/bench_brochure.py data/properties/sobha-neopolis luxury_landscape
    python scripts/bench_brochure.py data/properties/brigade-citrine --all   # every format of its category
    python scripts/bench_brochure.py data/properties/e128 --every            # all nine templates

No API key, no network: the Vision Agent's answers are read from
`analyses.json` beside the property (filename -> category, scene tag,
scores) when it exists, and every model call is refused so the offline
copywriter writes the book. It runs the real planner, composer, renderer,
Chromium, the rendered-page audit and the repair loop, and prints what the
quality gate saw.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MLFLOW_ENABLED", "false")
os.environ.setdefault("IMAGE_GEN_PROVIDER", "mock")


def offline_analyses(property_dir: Path, images) -> dict:
    """The Vision Agent's answers from analyses.json when it exists; the
    offline reading of the uploads (vision_offline) for anything else."""
    from src.agents.brochure import vision_offline
    from src.data.schemas import ImageAnalysis

    out = {}
    path = property_dir / "analyses.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        for image in images:
            entry = raw.get(image.filename)
            if entry:
                out[image.image_id] = ImageAnalysis(image_id=image.image_id, **entry)
    missing = [i for i in images if i.image_id not in out]
    out.update(vision_offline.run(missing))
    return out


def run(property_dir: Path, template: str, out_root: Path) -> dict:
    from src.agents.brochure import catalog, palettes, production
    from src.data.ingestion import discover_images, load_property_input

    started = time.time()
    prop = load_property_input(property_dir)
    spec = catalog.resolve(template, prop.category)
    images = discover_images(property_dir)
    analyses = offline_analyses(property_dir, images)
    out = out_root / f"{property_dir.name}__{spec.id}"
    out.mkdir(parents=True, exist_ok=True)
    images, _ = production.prepare_assets(images, analyses, out / "normalised", spec.id)
    palette = palettes.with_brand(palettes.for_category(spec.category), prop.brand_color)
    result = production.produce(prop, images, analyses, spec, palette, out, use_model=False)
    rep = result.report
    return {
        "template": spec.name, "property": prop.name, "pdf": str(result.pdf_path),
        "pages_count": len(result.book.pages), "passed": rep.passed,
        "mean_whitespace": round(rep.mean_whitespace(), 3),
        "defects": rep.defects[:20], "advisories": rep.advisories[:20],
        "repairs": result.repairs, "notes": result.book.notes,
        "seconds": round(time.time() - started, 1),
        "pages": [f"{p.number:02d} {p.kind:<11} open={p.whitespace:.2f} "
                  f"fill={','.join(f'{k}:{v:.2f}' for k, v in p.fill.items())} "
                  + "; ".join(d.split(": ", 1)[1] for d in p.defects) for p in rep.pages],
    }


def catalog_category(value) -> str:
    from src.agents.brochure.catalog import normalise_category

    return normalise_category(value) or "mid"


def main() -> int:
    import argparse

    from src.agents.brochure.catalog import TEMPLATES

    parser = argparse.ArgumentParser()
    parser.add_argument("property_dir")
    parser.add_argument("template", nargs="?", default="luxury_landscape")
    parser.add_argument("--all", action="store_true", help="every format of the property's category")
    parser.add_argument("--every", action="store_true", help="all nine templates")
    parser.add_argument("--out", default="data/generated/bench")
    args = parser.parse_args()
    if args.every:
        templates = list(TEMPLATES)
    elif args.all:
        from src.data.ingestion import load_property_input

        category = catalog_category(load_property_input(Path(args.property_dir)).category)
        templates = [t for t in TEMPLATES if t.startswith(category)]
    else:
        templates = [args.template]
    worst = 0
    for template in templates:
        report = run(Path(args.property_dir), template, Path(args.out))
        print(json.dumps({k: v for k, v in report.items() if k != "pages"}, indent=1))
        for page in report.get("pages", []):
            print("   ", page)
        worst = max(worst, 0 if report.get("passed") else 1)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
