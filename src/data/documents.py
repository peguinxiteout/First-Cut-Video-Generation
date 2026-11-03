"""Supporting documents for a property: PDF, Word, and plain text.

Optional: files placed in data/properties/<id>/documents/ (by the API or
by hand). Their text is extracted here and handed to the Property
Intelligence Agent, which turns it into facts, specifications, and legal
details for the brochure.

Two bounds keep this safe and affordable:
* size caps per document and in total, so a 200-page PDF can't blow the
  model's context or the free-tier token budget;
* extraction never raises. A corrupt or scanned file produces a note
  ("no extractable text") instead of failing the whole brochure.

Document text is DATA, not instructions — Property Intelligence's prompt
says so explicitly, since an uploaded file could contain text that reads
like an instruction to the model.
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".csv"}
MAX_CHARS_PER_DOCUMENT = 12_000
MAX_CHARS_TOTAL = 30_000
MAX_PDF_PAGES = 40
MAX_SHEET_ROWS = 300


@dataclass
class DocumentText:
    filename: str
    kind: str
    text: str
    truncated: bool = False
    note: str = ""


def _pdf(path: Path) -> str:
    import pdfplumber

    parts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages[:MAX_PDF_PAGES], start=1):
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(f"[page {i}]\n{text}")
    return "\n\n".join(parts)


def _docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(dict.fromkeys(cells)))  # merged cells repeat; dedupe in order
    return "\n".join(parts)


def _csv(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows = [" | ".join(c.strip() for c in r if c.strip()) for r in csv.reader(io.StringIO(raw))]
    return "\n".join(r for r in rows[:MAX_SHEET_ROWS] if r)


_EXTRACTORS = {".pdf": ("pdf", _pdf), ".docx": ("word", _docx), ".csv": ("csv", _csv),
               ".txt": ("text", lambda p: p.read_text(encoding="utf-8", errors="replace")),
               ".md": ("text", lambda p: p.read_text(encoding="utf-8", errors="replace"))}


def extract(path: Path) -> DocumentText:
    kind, fn = _EXTRACTORS.get(path.suffix.lower(), ("unsupported", None))
    if fn is None:
        return DocumentText(path.name, kind, "", note=f"Unsupported file type {path.suffix}; skipped.")
    try:
        text = fn(path).strip()
    except Exception as exc:  # corrupt / password-protected / unexpected layout
        logger.warning("Could not read %s: %s", path.name, exc)
        return DocumentText(path.name, kind, "", note=f"Could not read {path.name}: {exc.__class__.__name__}.")
    if not text:
        note = ("No extractable text (it may be a scanned image; add a text version for its details to be used)."
                if kind == "pdf" else "No text found.")
        return DocumentText(path.name, kind, "", note=f"{path.name}: {note}")
    truncated = len(text) > MAX_CHARS_PER_DOCUMENT
    return DocumentText(path.name, kind, text[:MAX_CHARS_PER_DOCUMENT], truncated=truncated,
                        note=f"{path.name}: only the first part was used (very long document)." if truncated else "")


def discover_documents(property_dir: Path) -> list[Path]:
    doc_dir = property_dir / "documents"
    if not doc_dir.is_dir():
        return []
    return sorted(p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def load_documents(property_dir: Path) -> tuple[str, list[str]]:
    """Returns (combined text for the model, human-readable notes)."""
    blocks, notes, used = [], [], 0
    for path in discover_documents(property_dir):
        doc = extract(path)
        if doc.note:
            notes.append(doc.note)
        if not doc.text:
            continue
        remaining = MAX_CHARS_TOTAL - used
        if remaining <= 0:
            notes.append(f"{doc.filename}: skipped (document budget reached).")
            continue
        text = doc.text[:remaining]
        used += len(text)
        blocks.append(f"=== {doc.filename} ({doc.kind}) ===\n{text}")
    return "\n\n".join(blocks), notes
