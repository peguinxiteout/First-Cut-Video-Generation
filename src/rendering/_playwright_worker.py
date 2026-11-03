#!/usr/bin/env python
"""
Standalone worker that does the actual Playwright rendering. Run via
subprocess (see render_pdf_and_screenshot() in renderer.py) — never
imported directly — because Playwright's sync API needs to launch a
subprocess of its own to run the browser, and that fails with a raw
`NotImplementedError` on Windows whenever asyncio's SelectorEventLoop is
already governing the calling thread. Streamlit (built on Tornado) forces
exactly that on Windows, so calling Playwright's sync API directly from
inside a Streamlit app breaks there — no amount of code change *within
that process* fixes it, since it's a property of the event loop already
running. A brand new OS process sidesteps this entirely: it gets Python's
normal default event loop (ProactorEventLoop on Windows), with none of the
parent process's asyncio state.

Optional 4th CLI arg: a Playwright browser "channel" (e.g. "msedge" or
"chrome") to use an already-installed SYSTEM browser instead of Playwright's
own downloaded Chromium build. This matters on locked-down Windows machines
running Application Control policies (WDAC/AppLocker) that block execution
of new/unsigned binaries — `playwright install chromium` downloads and
later launches an executable those policies often reject outright, while
system-installed Edge is virtually always already allowed. See
PLAYWRIGHT_BROWSER_CHANNEL in .env.example.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_AUDIT_JS = Path(__file__).resolve().parent / "brochure_audit.js"


# CSS px across an A4 sheet at scale 1, both ways round.
A4_CSS_SHORT = 794             # portrait width / landscape height
A4_CSS_LONG = 1123             # portrait height / landscape width

# Fallback only: the page size is MEASURED from the rendered document (see
# _page_size), because the stylesheet is the single source of truth for it
# and a constant here could silently disagree with a template.
DEFAULT_PAGE = (1240, 1754)


def _page_size(page) -> tuple[int, int]:
    """The authored size of one brochure page, read from the document.

    Templates are designed on different sheets — Signature on A4 portrait,
    the landscape templates on A4 landscape — and the viewport, the PDF
    scale and the per-page clipping all have to agree with whichever one
    this document uses. Measuring it here means a new template declares
    its page once, in its own CSS, and everything downstream follows."""
    try:
        size = page.evaluate(
            "() => { const el = document.querySelector('.page');"
            "        if (!el) return null;"
            "        const r = el.getBoundingClientRect();"
            "        return [Math.round(r.width), Math.round(r.height)]; }")
        if size and size[0] > 200 and size[1] > 200:
            return int(size[0]), int(size[1])
    except Exception:  # noqa: BLE001 - a measurement failure is not a render failure
        pass
    return DEFAULT_PAGE


PT_TO_CSS_PX = 96.0 / 72.0


def _trim_size(page) -> "tuple[float, float] | None":
    """The sheet this template is printed on, in POINTS, read from its CSS.

    Declared once, as `--trim-w-pt` / `--trim-h-pt` on :root, beside the `.page`
    rule that gives the authored pixel size — so a template describes its
    own paper and nothing here has to know which templates exist.

    This replaced `format="A4"`. That worked only while every page in the
    project shared A4's proportions; the reference brochures do not —
    Sobha is 1224 x 979pt (ratio 1.25) and Brigade 794 x 510pt (1.556) —
    and asking Chromium for A4 either crops such a page or distorts it."""
    try:
        size = page.evaluate(
            "() => { const s = getComputedStyle(document.documentElement);"
            "        const w = parseFloat(s.getPropertyValue('--trim-w-pt'));"
            "        const h = parseFloat(s.getPropertyValue('--trim-h-pt'));"
            "        return (w > 0 && h > 0) ? [w, h] : null; }")
        if size:
            return float(size[0]), float(size[1])
    except Exception:  # noqa: BLE001 - a measurement failure is not a render failure
        pass
    return None


def _pdf_scale(page_w: int, page_h: int, trim: "tuple[float, float] | None" = None) -> float:
    """What to scale the authored layout by so it lands exactly on its sheet.

    Exporting the authored pixel size directly produces a sheet that
    matches no real paper and cannot be printed — 930x1316pt for the
    portrait page. The trim the template declares is the paper; this maps
    one onto the other. With no declared trim the old A4 assumption
    stands, which is right for the two pages that are A4."""
    if trim:
        return round((trim[0] * PT_TO_CSS_PX) / page_w, 4)
    target = A4_CSS_LONG if page_w > page_h else A4_CSS_SHORT
    return round(target / page_w, 4)

def _bleed_sheet(page) -> "tuple[int, int] | None":
    """Switch the document into bleed mode and report the sheet size.

    The size is read back from the document rather than computed here, for
    the same reason the trim size is: the stylesheet is the only place a
    template's sheet is declared, and a constant in this file could
    silently disagree with it."""
    size = page.evaluate(
        "() => { document.documentElement.classList.add('bleed');"
        "        const el = document.querySelector('.sheet');"
        "        if (!el) { document.documentElement.classList.remove('bleed'); return null; }"
        "        const r = el.getBoundingClientRect();"
        "        return [Math.round(r.width), Math.round(r.height)]; }")
    if not size or size[0] < 200 or size[1] < 200:
        return None
    return int(size[0]), int(size[1])


def _clear_bleed(page) -> None:
    page.evaluate("() => document.documentElement.classList.remove('bleed')")


OVERVIEW_SCALE = 0.5          # the stacked image is a preview/fallback, not the judged artefact
OVERVIEW_MAX_HEIGHT = 20000   # PNG sanity bound for very long brochures


def _assemble_overview(shots, screenshot_path) -> None:
    """Stack the per-page captures into the single overview screenshot that
    the UI preview and the critic's fallback still expect. Done on the CPU
    because asking the browser for one giant full-page screenshot is the
    expensive part."""
    if not shots:
        return
    try:
        from PIL import Image

        images = [Image.open(s).convert("RGB") for s in shots]
        width = max(i.width for i in images)
        scale = OVERVIEW_SCALE
        total = int(sum(i.height for i in images) * scale)
        if total > OVERVIEW_MAX_HEIGHT:
            scale *= OVERVIEW_MAX_HEIGHT / total
        sheet = Image.new("RGB", (int(width * scale), int(sum(i.height for i in images) * scale)), "white")
        y = 0
        for img in images:
            resized = img.resize((int(img.width * scale), int(img.height * scale)))
            sheet.paste(resized, (0, y))
            y += resized.height
        sheet.save(screenshot_path)
    except Exception:  # noqa: BLE001 - the per-page shots are what matter
        pass


def _digital_frames(shots) -> None:
    """A 16:9 (digital presentation) page is delivered as 1920 x 1080 px."""
    try:
        from PIL import Image

        for shot in shots:
            with Image.open(shot) as im:
                if abs(im.width / im.height - 16 / 9) > 0.02 or im.size == (1920, 1080):
                    continue
                frame = im.convert("RGB").resize((1920, 1080), Image.LANCZOS)
            frame.save(shot)
    except Exception:  # noqa: BLE001 - the PDF is the deliverable
        pass


def main() -> int:
    if len(sys.argv) not in (4, 5, 6):
        print(
            "Usage: _playwright_worker.py <html_path> <pdf_path> <screenshot_path> "
            "[browser_channel] [pages|measure]",
            file=sys.stderr,
        )
        return 2

    html_path, pdf_path, screenshot_path = (Path(p) for p in sys.argv[1:4])
    channel = sys.argv[4] if len(sys.argv) >= 5 and sys.argv[4] else None
    # MEASURE mode: the audit, the reading PDF, and nothing else. The
    # workflow renders a brochure several times and only the last pass
    # produces files anybody looks at; on a long book the per-page images
    # and the second, bleed-sized PDF are most of the cost of a pass.
    capture_pages = not (len(sys.argv) >= 6 and sys.argv[5] == "measure")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        print(f"playwright is not installed: {exc}", file=sys.stderr)
        return 3

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            launch_kwargs = {"channel": channel} if channel else {}
            # An explicit binary path wins over both. Two real cases need
            # it: a machine whose policy forbids Playwright's own download
            # but already has a managed Chrome/Chromium, and an
            # environment where Playwright's pinned build number and the
            # installed browser disagree (the launcher then reports
            # "Executable doesn't exist at .../chromium-<n>" even though a
            # perfectly good browser is sitting next to it).
            binary = os.environ.get("PLAYWRIGHT_CHROMIUM_PATH", "").strip()
            if binary:
                launch_kwargs = {"executable_path": binary}
            browser = p.chromium.launch(**launch_kwargs)
            # Page images at twice screen resolution: sharp enough for the
            # critic and the preview, and for the 1920 x 1080 digital export.
            page = browser.new_page(viewport={"width": DEFAULT_PAGE[0], "height": DEFAULT_PAGE[1]},
                                    device_scale_factor=2 if capture_pages else 1)
            page.goto(html_path.resolve().as_uri())
            page.wait_for_timeout(300)  # let webfonts/images settle
            # ...and then WAIT for them, rather than hoping 300ms was enough.
            # The audit measures type; measured in a fallback font before the
            # embedded faces decode, every panel came out a different height
            # from the one printed, and the repair loop fixed the wrong pages.
            try:
                page.evaluate("""async () => {
                    await document.fonts.ready;
                    await Promise.all(Array.from(document.images).map(img => img.complete
                        ? Promise.resolve() : new Promise(r => { img.onload = img.onerror = r; })));
                    return true; }""")
            except Exception:  # noqa: BLE001 - a wait failure is not a render failure
                pass
            # Now that the document is loaded, size the viewport to the page
            # the TEMPLATE declares rather than to an assumption. A viewport
            # narrower than the page gives every element a horizontal
            # scrollbar's worth of offset, which the layout audit then
            # measures as the content sitting off the page.
            page_w, page_h = _page_size(page)
            if (page_w, page_h) != DEFAULT_PAGE:
                page.set_viewport_size({"width": page_w, "height": page_h})
                page.wait_for_timeout(120)
            # IMPORTANT: width/height here MUST match the .page CSS rule's
            # pixel dimensions exactly (see styles.css). Using format="A4"
            # instead renders each PDF page at ~794x1123px (A4 at 96 CSS
            # dpi) while our .page divs are sized 1240x1754px — every div's
            # content then overflows a physical A4 page by ~1.5x, spilling
            # onto a second, mostly-blank page. That's what caused "every
            # other page is blank": a page-size unit mismatch, not a
            # content bug.
            # Emit a PRINT-STANDARD A4 page. The layout is authored at
            # 1240x1754 CSS px; exporting those dimensions directly produced
            # a 930x1316pt page (12.9 x 18.3 in) that matches no sheet size
            # and cannot be printed. A4 holds 794 CSS px at scale 1, so
            # scaling by 794/1240 maps the layout exactly onto A4 (the two
            # aspect ratios agree to three decimal places).
            trim = _trim_size(page)
            scale = _pdf_scale(page_w, page_h, trim)
            if not capture_pages:
                pass  # MEASURE mode: the audit is all the repair loop needs
            elif trim:
                # In CSS px, not pt: Chromium's pdf() rejects a fractional
                # "1224.0pt" outright.
                page.pdf(path=str(pdf_path),
                         width=f"{trim[0] * PT_TO_CSS_PX:.0f}px",
                         height=f"{trim[1] * PT_TO_CSS_PX:.0f}px",
                         scale=scale, print_background=True,
                         margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
            else:
                page.pdf(path=str(pdf_path), format="A4", landscape=page_w > page_h,
                         scale=scale, print_background=True,
                         margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})

            # ...and a second, PRINT-READY file beside it.
            #
            # The reading copy above is exactly A4 with artwork running to
            # the edge, which is the one thing a printer cannot use: a
            # guillotine is accurate to about a millimetre, so it either
            # cuts into the photograph or leaves a white sliver down the
            # side. The print file carries 3mm of bleed on every edge and
            # crop marks in a 7mm slug outside it, on a correspondingly
            # larger sheet.
            #
            # One class on <html> switches the stylesheets between the two
            # — see the `.bleed` rules in each template — so both files
            # come from one render and cannot drift apart.
            try:
                sheet = _bleed_sheet(page) if capture_pages else None
                if sheet:
                    sheet_w, sheet_h = sheet
                    page.set_viewport_size({"width": sheet_w, "height": sheet_h})
                    page.wait_for_timeout(150)
                    print_path = pdf_path.with_name(pdf_path.stem + "-print.pdf")
                    page.pdf(path=str(print_path),
                             width=f"{sheet_w * scale}px", height=f"{sheet_h * scale}px",
                             scale=scale, print_background=True,
                             margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
                _clear_bleed(page)
                page.set_viewport_size({"width": page_w, "height": page_h})
                page.wait_for_timeout(120)
            except Exception as bleed_exc:  # noqa: BLE001 - the reading copy is the deliverable
                print(f"print-ready PDF not produced: {bleed_exc}", file=sys.stderr)
            # Audit FIRST: it reports each page's rect, which is then used to
            # capture pages by clipping.
            try:
                source = _AUDIT_JS.read_text(encoding="utf-8")
                report = page.evaluate("() => {" + source + "\nreturn auditBrochure(document, window); }")
            except Exception as audit_exc:  # noqa: BLE001
                report = {"error": str(audit_exc)}
            Path(str(screenshot_path) + ".layout.json").write_text(json.dumps(report), encoding="utf-8")

            # One capture per page, by CLIP. Previously this took a full-page
            # screenshot of the whole document AND a separate capture per
            # page: on a long brochure the stacked image is >25,000px tall,
            # which Chromium has to stitch together — slow and memory hungry
            # — and then every page was captured twice. That combination is
            # what made rendering time out. The stacked overview image other
            # code still expects is assembled from these on the CPU instead.
            pages_dir = Path(screenshot_path).parent / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            shots = []
            # By ELEMENT, not by clip. A clip outside the viewport needs
            # full_page=True, and without it Playwright captured page 1 and
            # silently failed on every page after it — the except below
            # swallowed sixteen errors a run. Nothing broke loudly: the PDF
            # is produced by page.pdf() and was always complete. What was
            # lost was everything that reads the page images — the contrast
            # audit (which was therefore only ever checking the COVER) and
            # the critic's vision judge, which saw a one-page brochure.
            #
            # An element screenshot scrolls the page into view and captures
            # a viewport-sized area, so it costs the same as the clip was
            # meant to and avoids the 25,000px full-page capture that made
            # rendering time out in the first place.
            sheets = page.query_selector_all("section.page") if capture_pages else []
            for index, element in enumerate(sheets, start=1):
                out = pages_dir / f"page_{index:02d}.png"
                try:
                    element.screenshot(path=str(out))
                    shots.append(out)
                except Exception:  # noqa: BLE001 - one missing page shot must not fail the render
                    pass
            if not shots:
                for rect in report.get("page_rects") or []:
                    out = pages_dir / f"page_{int(rect['index']):02d}.png"
                    try:
                        page.screenshot(path=str(out), full_page=True,
                                        clip={"x": rect["x"], "y": rect["y"],
                                              "width": rect["w"], "height": rect["h"]})
                        shots.append(out)
                    except Exception:  # noqa: BLE001
                        pass
            if not shots:  # no audit/rects: fall back to the old behaviour
                page.screenshot(path=str(screenshot_path), full_page=True)
            browser.close()
        _digital_frames(shots)
        _assemble_overview(shots, screenshot_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Playwright rendering failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
