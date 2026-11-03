"""Chromium, in its own process, turning a printed article into a PDF.

Separate from `_playwright_worker.py` because the two produce different
kinds of document. A brochure is a stack of pages the layout engine has
already decided the size of. An ARTICLE has no pages until Chromium makes
them: its length is not known until it is written, so the stylesheet says
where the text may break (templates/blog/default/print.css) and the
browser paginates.

Before printing, the page is measured in the browser: any element in the
article column that is wider than the column (an unbreakable URL, a table
that will not fit) is recorded in `<out.pdf>.audit.json`, which
src/rendering/blog_renderer.audit_pdf reads back.

Run as:  python _article_worker.py <source.html> <out.pdf> [channel]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_OVERFLOW_JS = """
() => {
  const out = [];
  const column = document.querySelector('.article');
  if (!column) return out;
  const limit = column.getBoundingClientRect().right + 2;
  document.querySelectorAll('.article *').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.right > limit) {
      const text = (el.innerText || el.tagName).trim().slice(0, 60);
      out.push(el.tagName.toLowerCase() + ': ' + text);
    }
  });
  return out.slice(0, 20);
}
"""

_FOOTER = (
    "<div style=\"width:100%;font-size:7pt;color:#8A8582;font-family:Montserrat,Helvetica,Arial,sans-serif;"
    "padding:0 26mm;display:flex;justify-content:space-between;align-items:center;\">"
    "<span style=\"max-width:70%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;\" class='title'></span>"
    "<span>Page <span class='pageNumber'></span> of <span class='totalPages'></span></span></div>"
)


def main() -> int:
    source = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2])
    channel = sys.argv[3] if len(sys.argv) > 3 else ""

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        kwargs = {"channel": channel} if channel else {}
        browser = pw.chromium.launch(**kwargs)
        try:
            # 794px = A4 width at 96 dpi, so screen measurements match paper.
            page = browser.new_page(viewport={"width": 794, "height": 1123})
            page.goto(source.resolve().as_uri())
            page.wait_for_timeout(400)      # webfonts and the data-URI images
            try:
                page.wait_for_load_state("networkidle", timeout=4000)
            except Exception:               # noqa: BLE001 - nothing to wait for
                pass
            page.emulate_media(media="print")
            try:
                overflow = page.evaluate(_OVERFLOW_JS)
            except Exception:               # noqa: BLE001 - the audit is advisory
                overflow = []
            Path(str(pdf_path) + ".audit.json").write_text(json.dumps({"overflow": overflow}), encoding="utf-8")
            # Margins live in @page (the first page has no top margin), so
            # none are passed here — passing them would override both.
            page.pdf(path=str(pdf_path), format="A4", print_background=True,
                     prefer_css_page_size=True,
                     margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                     display_header_footer=True,
                     header_template="<span></span>",
                     footer_template=_FOOTER)
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
