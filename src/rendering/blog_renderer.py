"""
Blog rendering: a Markdown file (easy to edit), an HTML preview, and the
PRIMARY deliverable — a PDF that reads like a well-designed published web
article (templates/blog/default/).

The PDF is made by Chromium's paged media from `print.html`. Because an
article's length is not known until it is written, nothing is placed by
hand; instead this module shapes the HTML so the browser CAN paginate it
well:

  * every heading is wrapped with the block that follows it (`.keep`), so a
    heading is never stranded at the foot of a page;
  * tables are wrapped so long cells wrap instead of overflowing, short
    tables are kept whole, and long ones repeat their header row;
  * "> **Tip:** ..." blockquotes become designed callouts, "- [ ] ..."
    items become a checklist with boxes, numbered lists with a bold
    lead-in become step cards;
  * figures are never split across pages (print.css).

After rendering, `audit_pdf` reads the PDF back and reports page count,
raw markup that leaked into the text, headings left at a page bottom and
elements that overflowed (measured in the browser by _article_worker.py).
"""
from __future__ import annotations

import json
import re
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import markdown as markdown_lib
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.core.config import PROJECT_ROOT
from src.data.schemas import BlogContent, BlogImageAsset, BlogInput
from src.rendering.fonts import font_face_css

TEMPLATES_DIR = PROJECT_ROOT / "templates" / "blog"

AI_IMAGE_LABEL = "Illustration (AI-generated)"
_MD_EXTENSIONS = ["tables", "sane_lists"]
WORDS_PER_MINUTE = 220


def _data_uri(path: Path) -> str | None:
    import base64
    import mimetypes

    if not path.exists():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Markdown -> article HTML
# ---------------------------------------------------------------------------

_CALLOUT = re.compile(
    r"<blockquote>\s*<p>\s*<strong>\s*(Tip|Note|Verify|Important|Warning|Watch out|Plan for today|"
    r"Where to verify|Key point)\s*:?\s*</strong>\s*:?\s*(.*?)</blockquote>", re.S | re.I)
_KIND = {"tip": "tip", "note": "note", "verify": "verify", "where to verify": "verify", "important": "warn",
         "warning": "warn", "watch out": "warn", "plan for today": "note", "key point": "note"}


def _postprocess(html: str) -> str:
    # Callouts.
    def _callout(m: re.Match) -> str:
        label = m.group(1).strip()
        kind = _KIND.get(label.lower(), "note")
        body = m.group(2).strip()
        if not body.startswith("<p>"):
            body = "<p>" + body
        return (f'<aside class="callout callout--{kind}"><div class="callout-label">{escape(label)}</div>'
                f"<div class=\"callout-body\">{body}</div></aside>")

    html = _CALLOUT.sub(_callout, html)
    html = re.sub(r"<blockquote>", '<aside class="callout callout--note"><div class="callout-body">', html)
    html = html.replace("</blockquote>", "</div></aside>")
    # Checklists.
    html = re.sub(r"<li>\s*\[\s?\]\s*", '<li class="task"><span class="box" aria-hidden="true"></span>', html)
    html = re.sub(r"<ul>(\s*<li class=\"task\")", r'<ul class="checklist">\1', html)
    # Step cards: an ordered list whose items open with a bold lead-in.
    html = re.sub(r"<ol>(\s*<li>\s*<strong>)", r'<ol class="steps">\1', html)
    # Tables: wrapped, classed by length.
    def _table(m: re.Match) -> str:
        table = m.group(0)
        rows = table.count("<tr")
        cls = "table-wrap table-short" if rows <= 9 else "table-wrap table-long"
        cols = len(re.findall(r"<th[\s>]", table.split("</tr>")[0]))
        return f'<div class="{cls} cols-{min(cols, 5)}">{table}</div>'

    html = re.sub(r"<table>.*?</table>", _table, html, flags=re.S)
    # Long unbroken strings (URLs) get break opportunities.
    html = re.sub(r">(https?://[^<\s]{30,})<", lambda m: ">" + m.group(1).replace("/", "/​") + "<", html)
    return html


def _md(text: str) -> str:
    return _postprocess(markdown_lib.markdown(text, extensions=_MD_EXTENSIONS)) if text else ""


class _TopLevel(HTMLParser):
    """Splits an HTML fragment into its top-level elements."""

    VOID = {"br", "img", "hr", "input", "meta", "link"}

    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_starts = [0]
        for i, ch in enumerate(source):
            if ch == "\n":
                self.line_starts.append(i + 1)
        self.depth = 0
        self.start = None
        self.chunks: list[str] = []

    def _pos(self) -> int:
        line, col = self.getpos()
        return self.line_starts[line - 1] + col

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID:
            if self.depth == 0:
                pos = self._pos()
                end = self.source.index(">", pos) + 1
                self.chunks.append(self.source[pos:end])
            return
        if self.depth == 0:
            self.start = self._pos()
        self.depth += 1

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        self.depth -= 1
        if self.depth == 0 and self.start is not None:
            end = self.source.index(">", self._pos()) + 1
            self.chunks.append(self.source[self.start:end])
            self.start = None


def top_level_chunks(html: str) -> list[str]:
    parser = _TopLevel(html)
    parser.feed(html)
    parser.close()
    return [c for c in parser.chunks if c.strip()]


def keep_headings_with_next(html: str) -> str:
    """Wrap each <h3>/<h4> with the element after it, so print.css can keep
    them on the same page."""
    chunks = top_level_chunks(html)
    out, i = [], 0
    while i < len(chunks):
        chunk = chunks[i]
        if re.match(r"<h[34]\b", chunk) and i + 1 < len(chunks):
            out.append(f'<div class="keep">{chunk}{chunks[i + 1]}</div>')
            i += 2
            continue
        out.append(chunk)
        i += 1
    return "\n".join(out)


def _small_block(chunk: str) -> bool:
    """Small enough to travel with a heading without leaving a hole: a
    short table, a list of up to four items, or a callout."""
    if chunk.startswith('<div class="table-wrap table-short'):
        return chunk.count("<tr") <= 7
    if re.match(r"<(?:ul|ol)\b", chunk):
        return chunk.count("<li") <= 4
    return chunk.startswith("<aside")


def split_lead(html: str) -> tuple[str, str]:
    """(first block, the rest) — the first block travels with the H2."""
    chunks = top_level_chunks(html)
    if not chunks:
        return "", ""
    first, rest = chunks[0], chunks[1:]
    if re.match(r"<h[34]\b", first) and rest:
        first, rest = first + rest[0], rest[1:]
    # A one-line lead-in ("Here is what is on record:") belongs with what it
    # introduces — otherwise heading + lead-in end a page and the table
    # starts the next one.
    if rest and re.match(r"<p>", first) and len(re.sub(r"<[^>]+>", "", first).split()) <= 45 \
            and _small_block(rest[0]):
        first, rest = first + rest[0], rest[1:]
    return first, keep_headings_with_next("\n".join(rest))


# ---------------------------------------------------------------------------
# Markdown file
# ---------------------------------------------------------------------------

def render_markdown(content: BlogContent, images_by_id: dict[str, BlogImageAsset]) -> str:
    """Builds the full post as one Markdown string."""
    lines = [f"# {content.title}", ""]
    if content.meta_description:
        lines += [f"*{content.meta_description}*", ""]

    if content.cover_image_id and content.cover_image_id in images_by_id:
        img = images_by_id[content.cover_image_id]
        lines += [f"![{content.title}]({img.filename})", ""]
        if content.hero_is_ai_generated:
            caption = f"{content.hero_caption} — " if content.hero_caption else ""
            lines += [f"*{caption}{AI_IMAGE_LABEL}*", ""]

    if content.intro_markdown:
        lines += [content.intro_markdown, ""]
    if content.key_takeaways:
        lines += ["## Key takeaways", ""] + [f"- {t}" for t in content.key_takeaways] + [""]

    for section in content.sections:
        lines += [f"## {section.heading}", ""]
        if section.image_id and section.image_id in images_by_id:
            img = images_by_id[section.image_id]
            lines += [f"![{section.heading}]({img.filename})", ""]
        lines += [section.body_markdown, ""]

    if content.faqs:
        lines += ["## Frequently asked questions", ""]
        for faq in content.faqs:
            lines += [f"### {faq.question}", "", faq.answer_markdown, ""]

    if content.conclusion_markdown:
        if content.conclusion_heading:
            lines += [f"## {content.conclusion_heading}", ""]
        lines += [content.conclusion_markdown, ""]

    if content.cta_markdown:
        lines += [f"## {content.cta_heading or 'Your next step'}", "", content.cta_markdown, ""]

    if content.references:
        lines += ["## Sources and where to verify", ""]
        for i, ref in enumerate(content.references, 1):
            link = f"[{ref.title}]({ref.url})" if ref.url else ref.title
            lines += [f"{i}. {link}" + (f" — {ref.note}" if ref.note else "")]
        lines += [""]
    if content.tags:
        lines += ["Topics: " + ", ".join(content.tags), ""]
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# HTML (preview + print)
# ---------------------------------------------------------------------------

def _context(content: BlogContent, images_by_id: dict[str, BlogImageAsset], blog_input: Optional[BlogInput],
             md_text: str, break_before: frozenset[str] = frozenset()) -> dict:
    from src.agents.blog import archetypes, brief

    def image_uri(image_id: str | None) -> str | None:
        """Images embedded as data URIs so the HTML is self-contained."""
        if not image_id or image_id not in images_by_id:
            return None
        return _data_uri(Path(images_by_id[image_id].path))

    sections = []
    for i, s in enumerate(content.sections, 1):
        lead, rest = split_lead(_md(s.body_markdown))
        sections.append({"heading": s.heading, "anchor": f"s{i}", "lead_html": lead, "rest_html": rest,
                         "image_uri": image_uri(s.image_id), "format": s.format or "prose",
                         "break_before": s.heading in break_before})
    contacts = brief.contact_details(blog_input) if blog_input else {"url": "", "phone": "", "email": ""}
    brand = (blog_input.brand_name if blog_input else "") or ""
    words = len(re.sub(r"[#*|>\-\[\]()]", " ", md_text).split())
    label = archetypes.catalogue().get(content.archetype, {}).get("label", "Guide")
    author = content.author or (brief.author(blog_input) if blog_input else "Editorial team")
    initials = "".join(w[0] for w in author.split()[:2]).upper() or "E"
    return dict(
        content=content,
        intro_html=_md(content.intro_markdown),
        conclusion_html=_md(content.conclusion_markdown),
        cta_html=_md(content.cta_markdown),
        faqs=[{"question": f.question, "answer_html": _md(f.answer_markdown)} for f in content.faqs],
        sections_html=sections,
        cover_image_uri=image_uri(content.cover_image_id),
        ai_label=AI_IMAGE_LABEL,
        font_face_css=font_face_css(("woff2",), ("Montserrat", "Lora")),
        cta_url=contacts["url"], cta_phone=contacts["phone"], cta_email=contacts["email"],
        cta_url_label=re.sub(r"^https?://(?:www\.)?", "", contacts["url"]).rstrip("/"),
        site_name=brand or "Property Insights",
        brand_description=(blog_input.brand_description if blog_input else "") or "",
        project_name=(blog_input.project_name if blog_input else "") or "",
        archetype_label=label,
        author=author, author_initials=initials,
        published_on=content.published_on,
        reading_minutes=max(1, round(words / WORDS_PER_MINUTE)),
        show_toc=len(content.sections) >= 4,
        has_supplied_facts=bool(blog_input and any(f.strip() for f in blog_input.local_facts)),
    )


def render_blog_files(
    content: BlogContent,
    images_by_id: dict[str, BlogImageAsset],
    output_dir: Path,
    template: str = "default",
    blog_input: Optional[BlogInput] = None,
    break_before: frozenset[str] = frozenset(),
) -> tuple[Path, Path]:
    """Writes content.md, preview.html and print.html into output_dir and
    returns (markdown, preview) paths. The PDF is a separate step
    (render_blog_pdf) because it needs a browser."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md_text = render_markdown(content, images_by_id)
    markdown_path = output_dir / "content.md"
    markdown_path.write_text(md_text, encoding="utf-8")

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR / template)),
                      autoescape=select_autoescape(["html"]))
    ctx = _context(content, images_by_id, blog_input, md_text, frozenset(break_before))
    html_path = output_dir / "preview.html"
    html_path.write_text(env.get_template("base.html").render(**ctx), encoding="utf-8")
    (output_dir / "print.html").write_text(env.get_template("print.html").render(**ctx), encoding="utf-8")
    return markdown_path, html_path


def render_blog_pdf(output_dir: Path) -> Optional[Path]:
    """Turn the printed article into a PDF. Returns None when the browser
    could not do it (the caller decides whether that is fatal — for the
    blog workflow it is, because the PDF is the deliverable)."""
    source = output_dir / "print.html"
    if not source.exists():
        return None
    pdf_path = output_dir / "article.pdf"
    try:
        from src.rendering.renderer import render_article_pdf

        render_article_pdf(source, pdf_path)
    except Exception as error:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("Blog PDF not produced: %s", error)
        return None
    return pdf_path if pdf_path.exists() else None


# ---------------------------------------------------------------------------
# Reading the PDF back
# ---------------------------------------------------------------------------

def _last_line(page, footer_pt: float = 30.0, line_pt: float = 13.0) -> str:
    """The lowest line of body text on a page, found by position (the
    PDF's text order is not reading order), above the footer band. The
    first band that catches anything may only catch descenders, so the
    fullest of the next few bands is taken."""
    width, height = page.get_size()
    textpage = page.get_textpage()
    try:
        y = footer_pt
        while y < height * 0.6:
            if textpage.get_text_bounded(left=0, bottom=y, right=width, top=y + line_pt).strip():
                bands = [textpage.get_text_bounded(left=0, bottom=y + d, right=width, top=y + d + line_pt)
                         for d in range(0, 18, 3)]
                return " ".join(max(bands, key=lambda t: len(t.strip())).split())
            y += 3
        return ""
    finally:
        textpage.close()


def audit_pdf(pdf_path: Path, headings: Optional[list[str]] = None) -> dict:
    """What a reader would see that we would rather they did not: leaked
    markdown, a heading stranded at a page bottom, overflowing elements."""
    from src.agents.blog.normaliser import markup_leaks

    notes: list[str] = []
    pages = 0
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        pages = len(pdf)
        texts, last_lines = [], []
        for i in range(pages):
            page = pdf[i]
            textpage = page.get_textpage()
            texts.append(textpage.get_text_range())
            textpage.close()
            last_lines.append(_last_line(page))
            page.close()
        pdf.close()
        for leak in markup_leaks("\n".join(texts)):
            notes.append(f"Raw markup visible in the PDF: {leak}.")
        # Compared without whitespace: PDF text extraction drops or adds
        # spaces depending on the font's glyph mapping.
        wanted = {"".join(h.split()).lower(): h.strip() for h in headings or [] if h and h.strip()}
        for i, line in enumerate(last_lines[:-1], 1):
            heading = wanted.get("".join(line.split()).lower())
            if heading:
                notes.append(f"Heading “{heading}” sits at the bottom of page {i}.")
    except Exception as exc:  # noqa: BLE001 - the audit is advisory
        notes.append(f"PDF audit skipped: {exc}")
    sidecar = Path(str(pdf_path) + ".audit.json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            for item in data.get("overflow", [])[:5]:
                notes.append(f"Content wider than the column: {item}.")
        except Exception:  # noqa: BLE001
            pass
    return {"pages": pages, "notes": notes}
