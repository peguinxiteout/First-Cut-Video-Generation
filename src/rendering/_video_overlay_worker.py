"""Chromium, in its own process, turning overlay pages into transparent PNGs.

Separate from the brochure workers for the same reason they are separate from
Streamlit: Playwright's sync API cannot start a browser from inside an event
loop on Windows. Run as:

    python _video_overlay_worker.py <jobs.json> [channel]

`jobs.json` is a list of {"html", "png", "width", "height"}. One browser is
started for all of them. Each page is screenshotted with a transparent
background; what its text elements (those marked data-safe) measure in the
page is written to <jobs.json>.measure.json, so the caller can check them
against the safe zone.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_MEASURE_JS = """
() => Array.from(document.querySelectorAll('[data-safe]')).map(el => {
  const r = el.getBoundingClientRect();
  return {x: r.x, y: r.y, w: r.width, h: r.height, text: (el.innerText || '').trim().slice(0, 80)};
})
"""


def main() -> int:
    jobs_path = Path(sys.argv[1])
    channel = sys.argv[2] if len(sys.argv) > 2 else ""
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))

    from playwright.sync_api import sync_playwright

    measured = []
    with sync_playwright() as pw:
        kwargs = {"channel": channel} if channel else {}
        binary = os.environ.get("PLAYWRIGHT_CHROMIUM_PATH", "").strip()
        if binary:
            kwargs = {"executable_path": binary}
        browser = pw.chromium.launch(**kwargs)
        try:
            for job in jobs:
                page = browser.new_page(viewport={"width": job["width"], "height": job["height"]})
                page.goto(Path(job["html"]).resolve().as_uri())
                try:
                    page.evaluate("async () => { await document.fonts.ready; return true; }")
                except Exception:               # noqa: BLE001 - a font wait that fails is not a render failure
                    pass
                rects = page.evaluate(_MEASURE_JS)
                page.screenshot(path=job["png"], omit_background=True)
                measured.append({"png": job["png"], "rects": rects})
                page.close()
        finally:
            browser.close()
    Path(str(jobs_path) + ".measure.json").write_text(json.dumps(measured), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
