"""Check this machine can run a demo — BEFORE the client is watching.

    python scripts/preflight.py

Verifies, in order of how likely it is to embarrass you live:
  1. Python version and the installed packages
  2. Playwright's browser (the single most common failure)
  3. API keys, and what degrades if one is missing
  4. bundled fonts, writable data directory, database
  5. an actual end-to-end render, offline, with no API calls

Exit code 0 = safe to demo. 1 = something will fail or degrade; every
line says what to do about it.
"""
from __future__ import annotations

import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status}] {name}{(' — ' + detail) if detail else ''}")


def check_python() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 12):
        record(OK, f"Python {major}.{minor}")
    else:
        record(FAIL, f"Python {major}.{minor}", "3.12 is required; recreate the venv with py -3.12")


def check_packages() -> None:
    missing = []
    for module, hint in [("streamlit", "UI"), ("jinja2", "templates"), ("PIL", "images"),
                         ("playwright", "PDF rendering"), ("langgraph", "workflows"),
                         ("pdfplumber", "PDF documents"), ("docx", "Word"),
                         ("huggingface_hub", "AI images")]:
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{module} ({hint})")
    if missing:
        record(FAIL, "Python packages", f"missing: {', '.join(missing)} — run pip install -r requirements.txt")
    else:
        record(OK, "Python packages")


def check_browser() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        record(FAIL, "Playwright", "not installed — pip install -r requirements.txt")
        return
    from src.core.config import settings
    try:
        binary = os.environ.get("PLAYWRIGHT_CHROMIUM_PATH", "").strip()
        with sync_playwright() as p:
            if binary:
                kwargs = {"executable_path": binary}
            else:
                kwargs = {"channel": settings.playwright_browser_channel} if settings.playwright_browser_channel else {}
            browser = p.chromium.launch(**kwargs)
            browser.close()
        record(OK, "Browser (Chromium)",
               binary or settings.playwright_browser_channel or "bundled")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Browser (Chromium)",
               f"{type(exc).__name__} — run: python -m playwright install chromium "
               f"(or set PLAYWRIGHT_BROWSER_CHANNEL=msedge in .env)")


def check_keys() -> None:
    from src.core.config import settings
    if settings.gemini_api_key:
        record(OK, "GEMINI_API_KEY", "text and vision agents ready")
    else:
        record(WARN, "GEMINI_API_KEY", "not set — every agent falls back to its offline path "
                                     "(no picture reading by a model, copy from the copy bank, "
                                     "measured evaluation only). Add it to .env for full quality")
    record(OK if settings.openrouter_api_key else WARN, "OPENROUTER_API_KEY",
           "fallback ready" if settings.openrouter_api_key else "no fallback if Gemini rate-limits mid-demo")
    has_cloudflare = bool(settings.cloudflare_account_id and settings.cloudflare_api_token)
    record(OK if has_cloudflare else WARN, "Image generation",
           "Cloudflare Workers AI ready" if has_cloudflare
           else "not configured — no supporting AI visuals in brochures and no blog hero image; "
                "placeholders are never printed (see CLOUDFLARE_ACCOUNT_ID in .env.example)")

    # Blog research tools. Most need no account at all, so this is usually
    # good news — and when it is not, it says which one is missing rather
    # than leaving thin posts unexplained.
    try:
        from src.tools import available_tools

        tools = available_tools(settings)
        ready = [t["tool"] for t in tools if t["available"]]
        missing = [t for t in tools if not t["available"]]
        if not settings.tools_enabled:
            record(WARN, "Blog research", "turned off (TOOLS_ENABLED=false)")
        elif len(ready) >= 6:
            detail = f"{len(ready)} of {len(tools)} tools ready"
            if missing:
                detail += " — optional: " + "; ".join(t["reason"] for t in missing)
            record(OK, "Blog research", detail)
        else:
            record(WARN, "Blog research",
                   "only %d tool(s) ready: %s" % (len(ready),
                                                  "; ".join(t["reason"] for t in missing)))
    except Exception as exc:  # noqa: BLE001 - preflight must never crash
        record(WARN, "Blog research", f"could not be checked: {exc}")


def check_assets() -> None:
    from src.rendering.fonts import FONTS_DIR, font_face_css
    faces = font_face_css().count("@font-face")
    record(OK if faces >= 10 else FAIL, "Bundled fonts",
           f"{faces} faces from {FONTS_DIR.name}/" if faces >= 10 else "font files missing from assets/fonts")


def check_storage() -> None:
    from src.core.config import settings
    try:
        data_dir = settings.data_dir_resolved
        (data_dir / ".preflight").write_text("ok", encoding="utf-8")
        (data_dir / ".preflight").unlink()
        from src.data import blog_repository, repository
        repository.init_db()
        blog_repository.init_db()
        record(OK, "Data directory and database", str(data_dir))
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Data directory and database", f"{type(exc).__name__}: {exc}")


def check_render() -> None:
    """Make a real brochure end to end — the form, the planner, the
    templates, fonts, the measure-and-repair loop, the PDF — with no API
    calls, so this works even with no keys."""
    import tempfile

    from PIL import Image

    from src.agents.brochure import catalog, palettes, production, vision_offline
    from src.data import property_form
    from src.data.ingestion import discover_images, load_property_input

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            uploads = {"photo": []}
            for i, (w, h, colour) in enumerate(((1600, 1000, (120, 110, 100)),
                                                 (1000, 1400, (90, 120, 140)),
                                                 (1600, 900, (150, 130, 90)))):
                image = tmp_path / f"photo_{i}.jpg"
                Image.new("RGB", (w, h), colour).save(image)
                uploads["photo"].append((image.name, image.read_bytes()))
            form = property_form.BrochureForm(
                name="Preflight Residences", city="Pune", category="Mid-Tier", orientation="Portrait",
                configurations=[property_form.ConfigRow(configuration="2 BHK", carpet_area="680 sq ft")],
                amenities="Clubhouse\nSwimming pool\nGym", phone="+91 98765 43210")
            folder = property_form.write_property(form, uploads, tmp_path / "properties")
            spec = catalog.resolve(property_form.template_for(form))
            prop = load_property_input(folder)
            images = discover_images(folder)
            analyses = vision_offline.run(images)
            result = production.produce(prop, images, analyses, spec,
                                        palettes.for_category(spec.category), tmp_path / "out",
                                        use_model=False, rounds=1)
            size = result.pdf_path.stat().st_size
            record(OK, "End-to-end brochure", f"{spec.name}: {len(result.book.pages)} pages, "
                                              f"PDF {size // 1024} KB, measured and checked")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "End-to-end brochure", f"{type(exc).__name__}: {exc}")


def main() -> int:
    print("Preflight — checking this machine can run a demo\n")
    check_python()
    check_packages()
    check_browser()
    check_keys()
    check_assets()
    check_storage()
    check_render()

    failures = [r for r in results if r[0] == FAIL]
    warnings = [r for r in results if r[0] == WARN]
    print()
    if failures:
        print(f"NOT READY — {len(failures)} blocking issue(s). Fix the FAIL lines above.")
        return 1
    if warnings:
        print(f"READY, with {len(warnings)} degraded feature(s) — see the WARN lines.")
        return 0
    print("READY — everything checks out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
