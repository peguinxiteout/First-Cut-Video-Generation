"""Shared rendering: HTML to PDF and page images in a separate Chromium
process, the PDF validity guard, the article PDF, and the QR code.

The brochure's HTML is written by src/rendering/brochure_renderer.py and the
blog's by src/rendering/blog_renderer.py; both come here to be printed."""
from __future__ import annotations

import base64
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path

from src.core.config import settings
from src.core.exceptions import RenderingError

_PLAYWRIGHT_WORKER = Path(__file__).resolve().parent / "_playwright_worker.py"
_ARTICLE_WORKER = Path(__file__).resolve().parent / "_article_worker.py"
_PLAYWRIGHT_WORKER_TIMEOUT_SECONDS = 120   # default; overridden by settings.playwright_timeout_seconds


def qr_code_data_uri(data: str, dark_color: str = "#000000") -> str | None:
    """Generates a real, scannable QR code on the fly (a base64 PNG data
    URI, so it works the same way for Playwright and any other renderer,
    no temp file to manage) — used by contact.html for the website/phone
    QR code the person can point a camera at. Returns None if `data` is
    empty or the qrcode dependency isn't available, rather than raising —
    a missing QR code should never fail the whole generation."""
    if not data:
        return None
    try:
        import qrcode
    except ImportError:
        return None
    img = qrcode.make(data, border=1)
    try:
        img = img.convert("RGBA")
        pixels = img.load()
        # Recolor from pure black to the template's own dark ink color so
        # the QR code doesn't look like a bolted-on generic graphic.
        target = tuple(int(dark_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        for y in range(img.height):
            for x in range(img.width):
                r, g, b, a = pixels[x, y]
                if (r, g, b) == (0, 0, 0):
                    pixels[x, y] = (*target, a)
    except Exception:  # noqa: BLE001 - recoloring is cosmetic; fall back to plain black/white if it fails
        pass
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_DIGIT_RUN = re.compile(r"(?<![\d.,])(\d{4,})(?![\d.,])")


def group_digits(value) -> str:
    """Thousands separators in a FACT, and nothing else.

    From the print review: "Areas are written '5789 sq. ft./ 537.81 sq. m.'
    with no thousands separator."

    Applied to the property's own data — the configuration, area and price
    a person entered in the form — where a number is a number and grouping
    it is presentation. It is not applied to prose.

    Decimals and already-grouped numbers are left alone, so running this
    twice changes nothing."""
    text = "" if value is None else str(value)
    return _DIGIT_RUN.sub(lambda m: f"{int(m.group(1)):,}", text)


def render_pdf_and_screenshot(html_path: Path, pdf_path: Path, screenshot_path: Path,
                              *, capture_pages: bool = True) -> None:
    """Turns the rendered HTML into a PDF and a full-page PNG screenshot
    (the screenshot is what the Critic agent's vision judge looks at).

    Runs the actual Playwright/Chromium work in a SEPARATE OS process
    (src/rendering/_playwright_worker.py) rather than in-process. This
    isn't just style: calling Playwright's sync API directly from inside
    an app like Streamlit breaks with a raw NotImplementedError on
    Windows, because Streamlit (built on Tornado) forces asyncio's
    SelectorEventLoop, which can't launch subprocesses on Windows — and
    Playwright's sync API needs to launch one to run the browser. A fresh
    process sidesteps this: it gets Python's normal default event loop,
    unaffected by whatever asyncio state the caller's process/thread is in.
    This also just works identically on every platform, not only Windows.

    `capture_pages=False` measures only: the layout audit is written, and
    the PDF, the print-ready PDF and the page images are not. The QA loop
    (src/agents/brochure/production.py) measures several times and only
    the last render produces files anybody looks at."""
    if not _PLAYWRIGHT_WORKER.exists():  # pragma: no cover
        raise RenderingError(f"Playwright worker script missing: {_PLAYWRIGHT_WORKER}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(_PLAYWRIGHT_WORKER),
                str(html_path),
                str(pdf_path),
                str(screenshot_path),
                settings.playwright_browser_channel,
                "pages" if capture_pages else "measure",
            ],
            capture_output=True,
            text=True,
            timeout=getattr(settings, "playwright_timeout_seconds", _PLAYWRIGHT_WORKER_TIMEOUT_SECONDS),
        )
    except subprocess.TimeoutExpired as exc:
        limit = getattr(settings, "playwright_timeout_seconds", _PLAYWRIGHT_WORKER_TIMEOUT_SECONDS)
        raise RenderingError(
            f"Playwright rendering timed out after {limit}s. Rendering scales with the number of pages "
            f"and the size of the photographs: try fewer or smaller images, or raise "
            f"PLAYWRIGHT_TIMEOUT_SECONDS in .env."
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RenderingError(f"Playwright rendering failed: {detail}")

    if capture_pages and not is_pdf_render_valid(pdf_path):
        raise RenderingError(
            "Playwright reported success but the PDF file is missing or unexpectedly "
            "small — treating this as a failed render rather than risking a broken or "
            "blank result reaching the user."
        )

    # The press condition this file is built for, written onto the
    # print-ready PDF. It is done HERE and not inside the worker: the
    # worker is launched as a bare script, so the project is not on its
    # import path and the import failed there silently — the file went out
    # saying nothing about the press it was built for, which is the thing
    # the review asked for. See src/rendering/prepress.py.
    if capture_pages:
        print_path = pdf_path.with_name(pdf_path.stem + "-print.pdf")
        if print_path.exists():
            from src.rendering import prepress

            prepress.apply_output_intent(print_path)


# A genuinely rendered multi-page brochure PDF is well above this; catches a
# process that "succeeded" (exit code 0) but wrote a near-empty/corrupt file
# — a rare edge case, but one the workflow should fail loudly on rather than
# silently hand a broken file to the Critic and then the user. This is a
# binary pass/fail rendering-pipeline guard, not a scored evaluation
# dimension (see the note at the top of src/evaluation/scoring.py for why
# "completeness" was removed as something that partially inflated the
# Quality Score — this check replaces its only legitimate purpose).
_MIN_VALID_PDF_BYTES = 5_000



def is_pdf_render_valid(pdf_path: Path) -> bool:
    return pdf_path.exists() and pdf_path.stat().st_size >= _MIN_VALID_PDF_BYTES


def render_article_pdf(html_path: Path, pdf_path: Path) -> Path:
    """A blog post as a printed article.

    Different from `render_pdf_and_screenshot` in one way that matters:
    the pages do not exist until Chromium makes them. A brochure's pages
    are decided by the layout engine and exported as authored; an
    article's length is not known until it is written, so the stylesheet
    says where it may break and the browser paginates. See
    src/rendering/_article_worker.py."""
    if not _ARTICLE_WORKER.exists():  # pragma: no cover
        raise RenderingError(f"Article worker script missing: {_ARTICLE_WORKER}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(_ARTICLE_WORKER), str(html_path), str(pdf_path),
         settings.playwright_browser_channel],
        capture_output=True, text=True,
        timeout=getattr(settings, "playwright_timeout_seconds", _PLAYWRIGHT_WORKER_TIMEOUT_SECONDS),
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RenderingError(f"Article rendering failed: {detail}")
    if not is_pdf_render_valid(pdf_path):
        raise RenderingError("Article PDF was produced but is not a readable PDF")
    return pdf_path
