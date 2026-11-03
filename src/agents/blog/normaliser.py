"""Markdown normaliser — fixes what is mechanically fixable in a draft.

Every formatting defect the evaluation found in the PDFs came from the
same place: model-written markdown that LOOKS fine as text and renders
badly. "- a - b - c" on one line renders as one bullet. A list typed
straight under a paragraph, with no blank line, renders as part of the
paragraph. An unclosed ** leaks into the PDF as two asterisks. A table
missing its |---| row renders as a line of pipes. None of that needs a
model to fix, and asking a model to fix it does not reliably work, so it
is fixed here, in code, every time, for both the LLM and the fallback
writer.

Each function takes markdown and returns (markdown, fixes) where fixes is
a list of plain-language notes for the validation report. What cannot be
fixed mechanically (an untraceable figure is fixable — by removal — but a
generic heading is not) is reported by the Editor's validator instead.

Pure functions, no I/O, no model: deterministic and unit-testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from src.agents.blog import blog_config

# ---------------------------------------------------------------------------
# Line classification
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s*(\S.*?)\s*#*\s*$")
_LIST = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+")
_TABLE = re.compile(r"^\s*\|.*\|\s*$|^\s*\|")
_QUOTE = re.compile(r"^\s*>")
_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _kind(line: str) -> str:
    if not line.strip():
        return "blank"
    if _HEADING.match(line):
        return "heading"
    if _SEPARATOR.match(line) and "|" in line:
        return "table"
    if _TABLE.match(line):
        return "table"
    if _LIST.match(line):
        return "list"
    if _QUOTE.match(line):
        return "quote"
    return "text"


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------

_ABBREVIATIONS = ("e.g.", "i.e.", "rs.", "approx.", "no.", "st.", "dr.", "mr.", "mrs.", "ms.", "vs.",
                  "etc.", "sq.", "ft.", "inc.", "ltd.", "pvt.", "co.", "jr.", "sr.", "u.s.", "a.m.", "p.m.")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[*]?[A-Z₹0-9])")


def split_sentences(text: str) -> list[str]:
    """Sentence split that survives "e.g.", "Rs. 50" and "1.2 km"."""
    text = " ".join(text.split())
    if not text:
        return []
    out: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        candidate = text[start:match.start()]
        tail = candidate.lower().rsplit(" ", 1)[-1]
        if any(tail.endswith(abbr) for abbr in _ABBREVIATIONS):
            continue
        out.append(text[start:match.end()].strip())
        start = match.end()
    rest = text[start:].strip()
    if rest:
        out.append(rest)
    return [s for s in out if s]


def words(text: str) -> list[str]:
    plain = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.findall(r"[A-Za-z0-9₹][\w'’.,%-]*", plain)


# ---------------------------------------------------------------------------
# Placeholders, links, contacts
# ---------------------------------------------------------------------------

def _placeholder_patterns() -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in blog_config.editorial().get("placeholder_patterns", [])]


def find_placeholders(text: str) -> list[str]:
    found = []
    for pattern in _placeholder_patterns():
        found.extend(m.group(0) for m in pattern.finditer(text or ""))
    return found


_MD_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]*)(?:\s+\"[^\"]*\")?\)")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BARE_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>()\]]+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PLACEHOLDER_MARK = "\u2063PH\u2063"
_PHONE = re.compile(r"(?<![\w₹])\+?\d[\d\s().-]{8,}\d(?![\w%])")


def _host(url: str) -> str:
    return re.sub(r"^(?:https?://)?(?:www\.)?", "", url.lower()).split("/")[0].strip(".")


def url_allowed(url: str, allowed: Iterable[str]) -> bool:
    url = url.strip()
    if not url or url == "#" or not re.match(r"^(?:https?://|www\.)", url, re.I):
        return False
    if find_placeholders(url):
        return False
    hosts = {_host(a) for a in allowed if a}
    return _host(url) in hosts or url.rstrip("/") in {a.rstrip("/") for a in allowed}


def _phone_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _is_phone(candidate: str) -> bool:
    digits = _phone_digits(candidate)
    return 10 <= len(digits) <= 13 and not re.search(r"\d\s?[-–]\s?\d{1,3}\b\s*(?:lakh|crore|km)", candidate)


# ---------------------------------------------------------------------------
# Normalisation passes
# ---------------------------------------------------------------------------

@dataclass
class NormaliseResult:
    text: str
    fixes: list[str] = field(default_factory=list)
    issues: list[tuple[str, str, str]] = field(default_factory=list)   # (code, severity, message)


def _note(fixes: list[str], message: str) -> None:
    if message not in fixes:
        fixes.append(message)


def preclean(md: str, fixes: list[str]) -> str:
    """HTML, code fences, invisible characters, odd bullets."""
    text = (md or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = text.replace(" ", " ").replace("​", "").replace("﻿", "")
    text = text.replace("\\n", "\n") if "\\n" in text and "\n" not in text else text
    if re.search(r"<\s*br\s*/?\s*>", text, re.I):
        text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.I)
        _note(fixes, "Converted HTML line breaks to markdown.")
    if re.search(r"</?(?:b|strong)>", text, re.I):
        text = re.sub(r"</?(?:b|strong)>", "**", text, flags=re.I)
    if re.search(r"</?(?:i|em)>", text, re.I):
        text = re.sub(r"</?(?:i|em)>", "*", text, flags=re.I)
    if re.search(r"<li>", text, re.I):
        text = re.sub(r"\s*<li>\s*", "\n- ", text, flags=re.I)
    if re.search(r"</?(?:p|div|span|ul|ol|li|h\d|section|article)[^>]*>", text, re.I):
        text = re.sub(r"</?(?:p|div|span|ul|ol|li|h\d|section|article)[^>]*>", "\n", text, flags=re.I)
        _note(fixes, "Removed raw HTML tags.")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    if "```" in text:
        text = re.sub(r"```[a-zA-Z]*\n?", "", text)
        _note(fixes, "Removed code fences.")
    if "`" in text:
        text = text.replace("`", "")
    if _MD_IMAGE.search(text):
        text = _MD_IMAGE.sub("", text)
        _note(fixes, "Removed inline image markup (images are placed by the layout).")
    return text


_BULLET_CHARS = "•◦▪▫●○■□►▶✓✔➤➢–—"


def fix_lines(text: str, fixes: list[str], allow_headings: bool = True) -> str:
    """Line-level repairs: bullet markers, collapsed bullets, inline
    numbered lists, heading syntax and levels, pseudo-headings."""
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        stripped = line.lstrip()

        # Checkbox characters -> markdown task items.
        if re.match(r"^[☐□❑]\s*", stripped):
            line = "- [ ] " + re.sub(r"^[☐□❑]\s*", "", stripped)
            stripped = line
        if re.match(r"^[-*+]\s*\[\s?[xX ]?\s?\]\s*", stripped):
            line = "- [ ] " + re.sub(r"^[-*+]\s*\[\s?[xX ]?\s?\]\s*", "", stripped)
            stripped = line

        # Unicode / alternative bullet markers.
        if stripped[:1] in _BULLET_CHARS and len(stripped) > 1 and stripped[1] in " \t":
            line = "- " + stripped[1:].strip()
            _note(fixes, "Normalised bullet markers.")
            stripped = line
        elif re.match(r"^[*+]\s+\S", stripped) and not stripped.startswith("**"):
            line = "- " + stripped[1:].strip()
            stripped = line
        elif re.match(r"^\d+\)\s+", stripped):
            line = re.sub(r"^(\d+)\)\s+", r"\1. ", stripped)
            stripped = line

        # Collapsed "• a • b • c" anywhere in a line.
        if stripped.count(" • ") >= 1 or (stripped.startswith("- ") and " • " in stripped):
            head, *items = re.split(r"\s*•\s+", stripped)
            head = head.strip()
            if head.startswith("- "):
                items = [head[2:]] + items
                head = ""
            if head:
                out.append(head)
                out.append("")
            out.extend(f"- {item.strip()}" for item in items if item.strip())
            _note(fixes, "Split bullets that had been collapsed onto one line.")
            continue

        # Collapsed "- a - b - c" (list item or "Intro: - a - b").
        separators = list(re.finditer(r"(?<=\S)\s+[-–—]\s+(?=[A-Z₹\[*(])", stripped))
        separators = [m for m in separators
                      if not (stripped[m.start() - 1:m.start()].isdigit()
                              and stripped[m.end():m.end() + 1].isdigit())]
        is_item = stripped.startswith("- ")
        intro_list = re.match(r"^(.{3,}?[:])\s*[-–—]\s+(?=\S)", stripped) and len(separators) >= 1
        if (is_item and len(separators) >= 1 and _looks_like_list(stripped, separators)) or intro_list:
            if intro_list:
                head, rest = re.match(r"^(.{3,}?[:])\s*[-–—]\s+(.*)$", stripped).groups()
                out.append(head.strip())
                out.append("")
                pieces = re.split(r"\s+[-–—]\s+(?=[A-Z₹\[*(])", rest)
            else:
                pieces = re.split(r"\s+[-–—]\s+(?=[A-Z₹\[*(])", stripped[2:])
            out.extend(f"- {p.strip()}" for p in pieces if p.strip())
            _note(fixes, "Split bullets that had been collapsed onto one line.")
            continue

        # Inline numbered list: "1. Do this 2. Then that 3. Finally".
        numbered = list(re.finditer(r"(?:(?<=\s)|^)(\d{1,2})[.)]\s+(?=[A-Z₹*])", stripped))
        if len(numbered) >= 2 and [int(m.group(1)) for m in numbered] == \
                list(range(int(numbered[0].group(1)), int(numbered[0].group(1)) + len(numbered))):
            head = stripped[:numbered[0].start()].strip()
            if head:
                out.append(head)
                out.append("")
            for i, m in enumerate(numbered):
                end = numbered[i + 1].start() if i + 1 < len(numbered) else len(stripped)
                out.append(f"{m.group(1)}. {stripped[m.end():end].strip()}")
            _note(fixes, "Split a numbered list that had been run into one line.")
            continue

        # Headings: "##Heading", bold inside, trailing colon, level.
        if re.match(r"^#{1,6}\S", stripped):
            stripped = re.sub(r"^(#{1,6})", r"\1 ", stripped)
            line = stripped
        heading = _HEADING.match(stripped)
        if heading:
            level, title = len(heading.group(1)), heading.group(2)
            title = re.sub(r"\*\*|__|\*", "", title).strip().rstrip(":").strip()
            if not allow_headings:
                out.append(title + ("" if title.endswith((".", "?", "!")) else ""))
                _note(fixes, "Removed a heading from a part of the post that should not have one.")
                continue
            new_level = 3 if level <= 3 else 4
            if new_level != level:
                _note(fixes, "Corrected heading levels inside sections.")
            out.append("#" * new_level + " " + title)
            continue

        # A line that is only bold text is a heading in disguise.
        pseudo = re.match(r"^\*\*([^*]{2,80}?)\*\*:?\s*$", stripped)
        if pseudo and allow_headings and len(pseudo.group(1).split()) <= 10:
            out.append("### " + pseudo.group(1).strip().rstrip(":"))
            _note(fixes, "Turned bold-only lines into real subheadings.")
            continue

        out.append(line)
    return "\n".join(out)


def _looks_like_list(line: str, separators: list) -> bool:
    """A dash between capitalised phrases is a collapsed list; a dash in a
    sentence ("the metro – when it opens – will help") is punctuation."""
    pieces = re.split(r"\s+[-–—]\s+(?=[A-Z₹\[*(])", line[2:])
    if len(pieces) < 2:
        return False
    short = sum(1 for p in pieces if len(p.split()) <= 14)
    return short == len(pieces) or len(pieces) >= 3


def separate_blocks(text: str, fixes: list[str]) -> str:
    """Blank lines between blocks of different kinds — the single most
    common reason a list rendered as part of a paragraph — and paragraph
    breaks where a model used one newline between paragraphs."""
    lines = text.split("\n")
    out: list[str] = []
    prev_kind = "blank"
    inserted = False
    for line in lines:
        kind = _kind(line)
        if kind == "blank":
            out.append("")
            prev_kind = "blank"
            continue
        if prev_kind != "blank":
            prev = out[-1]
            if kind != prev_kind:
                if prev_kind == "list" and kind == "text" and line.startswith("  "):
                    pass                                    # indented continuation
                elif prev_kind == "list" and kind == "text" and line.strip()[:1].islower():
                    out[-1] = prev + " " + line.strip()     # lazy continuation of the item
                    continue
                else:
                    out.append("")
                    inserted = True
            elif kind == "text" and _ends_sentence(prev):
                out.append("")                              # one newline between paragraphs
                inserted = True
            elif kind == "text":
                out[-1] = prev + " " + line.strip()         # soft-wrapped paragraph
                continue
            elif kind == "heading":
                out.append("")
        out.append(line)
        prev_kind = kind
    if inserted:
        _note(fixes, "Added the blank lines markdown needs between paragraphs, lists and tables.")
    joined = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")
    # A blank line between two items makes a "loose" list (each item a
    # paragraph, spaced like prose); items belong together.
    tight = re.sub(r"(?m)^((?:\s*(?:[-*+]|\d+[.)])\s+).*)\n\n(?=\s*(?:[-*+]|\d+[.)])\s+)", r"\1\n", joined)
    while tight != joined:
        joined = tight
        tight = re.sub(r"(?m)^((?:\s*(?:[-*+]|\d+[.)])\s+).*)\n\n(?=\s*(?:[-*+]|\d+[.)])\s+)", r"\1\n", joined)
    return tight


def _ends_sentence(line: str) -> bool:
    return bool(re.search(r"[.!?:;\"')\]*]\s*$", line.strip()))


def blocks_of(text: str) -> list[str]:
    return [b for b in re.split(r"\n\s*\n", text) if b.strip()]


def block_kind(block: str) -> str:
    first = block.split("\n", 1)[0]
    return _kind(first)


# --- tables ------------------------------------------------------------------

def _cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", row)]


def normalise_table(block: str, fixes: list[str]) -> str:
    """A well-formed GFM table, or a list if it was never really a table."""
    rows = [r for r in block.split("\n") if r.strip()]
    body = [r for r in rows if not _SEPARATOR.match(r)]
    if not body:
        _note(fixes, "Removed a table separator with no table.")
        return ""
    parsed = [_cells(r) for r in body]
    parsed = [[re.sub(r"\s*<br\s*/?>\s*", "; ", c) for c in r] for r in parsed]
    header = [re.sub(r"\*\*", "", c) for c in parsed[0]]
    width = len(header) if len(header) >= 2 else max(len(r) for r in parsed)
    data = [r for r in parsed[1:] if any(c.strip() for c in r)]
    if width < 2:
        _note(fixes, "Turned a one-column table into a list.")
        return "\n".join(f"- {r[0]}" for r in parsed if r and r[0])
    if not data:
        _note(fixes, "Removed a table that had a header but no rows.")
        return ""
    fixed = []
    for r in [header] + data:
        if len(r) < width:
            r = r + [""] * (width - len(r))
        elif len(r) > width:
            r = r[:width - 1] + [" ".join(r[width - 1:])]
        fixed.append(r)
    if len(rows) != len(body) + 1 or not (len(rows) > 1 and _SEPARATOR.match(rows[1])) \
            or any(len(p) != width for p in parsed):
        _note(fixes, "Repaired malformed table markup.")
    lines = ["| " + " | ".join(fixed[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(c if c else "—" for c in r) + " |" for r in fixed[1:]]
    return "\n".join(lines)


# --- inline markup -----------------------------------------------------------

def fix_inline(block: str, fixes: list[str]) -> str:
    """Stray emphasis and heading marks inside running text."""
    text = block
    if "__" in text:
        text = re.sub(r"__(\S[^_]*?\S|\S)__", r"**\1**", text)
    text = re.sub(r"\*{4,}", "", text)
    new_lines = []
    for line in text.split("\n"):
        if line.count("**") % 2:
            line = line.replace("**", "")
            _note(fixes, "Removed unbalanced bold markers that would have printed as asterisks.")
        body_only = re.sub(r"^\s*[-*+]\s", "", line).replace("**", "")
        if body_only.count("*") % 2:
            marker = re.match(r"^\s*[-*+]\s", line)
            head = marker.group(0) if marker else ""
            rest = line[len(head):]
            rest = re.sub(r"(?<!\*)\*(?!\*)", "", rest)
            line = head + rest
            _note(fixes, "Removed stray asterisks.")
        if re.search(r"(?<=\S)\s#{1,6}\s", line) and not _HEADING.match(line):
            line = re.sub(r"(?<=\S)\s#{1,6}\s", " ", line)
            _note(fixes, "Removed heading marks from inside a sentence.")
        if _SEPARATOR.match(line) and "|" in line:
            continue
        new_lines.append(line)
    text = "\n".join(new_lines)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])(?=\s|$)", r"\1", text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)
    return text


def normalise_units(text: str, fixes: list[str]) -> str:
    """One spelling per unit, so the same figure reads the same way
    everywhere: ₹, lakh, crore, sq ft, km, min."""
    original = text
    text = re.sub(r"\b(?:Rs\.?|INR)\s?(?=\d)", "₹", text)
    text = re.sub(r"₹\s+(?=\d)", "₹", text)
    text = re.sub(r"(?<=\d)\s?(?:lakhs|lacs|lac)\b", " lakh", text, flags=re.I)
    text = re.sub(r"(?<=\d)\s?(?:crores|Cr\.?|cr\.?)(?=\s|$|[,.;)])", " crore", text)
    text = re.sub(r"(?<=\d)\s?(?:sq\.?\s?ft\.?|sqft|sft|square\s+feet|square\s+foot)(?=[\s,.;)]|$)",
                  " sq ft", text, flags=re.I)
    text = re.sub(r"\bper\s+(?:sq\.?\s?ft\.?|sqft|sft|square\s+(?:feet|foot))(?=[\s,.;)]|$)", "per sq ft",
                  text, flags=re.I)
    text = re.sub(r"(?<=\d)\s?(?:sq\.?\s?m\.?|sqm|square\s+met(?:re|er)s?)(?=[\s,.;)]|$)", " sq m",
                  text, flags=re.I)
    text = re.sub(r"(?<=\d)\s?(?:kms|KMs|Kms|KM|Km)\b", " km", text)
    text = re.sub(r"(?<=\d)(km|min)\b", r" \1", text)
    text = re.sub(r"(?<=\d)\s?mins\b", " min", text)
    text = re.sub(r"(?<=\d)\s+%", "%", text)
    if text != original:
        _note(fixes, "Standardised units (₹, lakh/crore, sq ft, km, min).")
    return text


def replace_stock_phrases(text: str, fixes: list[str]) -> tuple[str, list[str]]:
    """Stock AI phrasing from configs/blog/editorial.yaml. Returns the text
    and the sentences dropped because the phrase carried the sentence."""
    phrases: dict = blog_config.editorial().get("stock_phrases", {}) or {}
    dropped: list[str] = []
    for phrase, replacement in sorted(phrases.items(), key=lambda kv: -len(kv[0])):
        before = r"(?<![\w-])" if phrase[:1].isalnum() else ""
        after = r"(?![\w-])" if phrase[-1:].isalnum() else ""
        pattern = re.compile(before + re.escape(phrase) + after, re.I)
        if not pattern.search(text):
            continue
        if replacement is None:
            text, removed = remove_sentences(text, lambda s, p=pattern: bool(p.search(s)))
            dropped.extend(removed)
        else:
            def _sub(m, rep=replacement):
                if not rep:
                    return ""
                return rep[:1].upper() + rep[1:] if m.group(0)[:1].isupper() else rep
            text = pattern.sub(_sub, text)
        # Capitalise a sentence that now starts lower-case after a removal.
        text = re.sub(r"(^|[.!?]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
        _note(fixes, "Rewrote stock AI phrasing.")
    text = re.sub(r"[ ]{2,}", " ", text)
    return text, dropped


def cap_bold(text: str, fixes: list[str], max_per_paragraph: int = 1, max_words: int = 8) -> str:
    """At most one bold phrase per paragraph and none longer than a short
    phrase; a bold lead-in label on a list item ("**Commute:** ...") is
    allowed, extra bold inside the item is not."""
    out_blocks = []
    changed = False
    for block in blocks_of(text):
        kind = block_kind(block)
        if kind == "table":
            new = re.sub(r"\*\*([^*]+)\*\*", r"\1", block)
        elif kind == "list":
            new_lines = []
            for line in block.split("\n"):
                lead = re.match(r"^(\s*(?:[-*+]|\d+[.)])\s+(?:\[ \]\s+)?)(\*\*[^*]{1,60}\*\*:?)(.*)$", line)
                if lead:
                    rest = re.sub(r"\*\*([^*]+)\*\*", r"\1", lead.group(3))
                    new_lines.append(lead.group(1) + lead.group(2) + rest)
                else:
                    new_lines.append(_limit_bold(line, 1, max_words))
            new = "\n".join(new_lines)
        elif kind == "heading":
            new = re.sub(r"\*\*([^*]+)\*\*", r"\1", block)
        else:
            new = _limit_bold(block, max_per_paragraph, max_words)
        changed |= new != block
        out_blocks.append(new)
    if changed:
        _note(fixes, "Reduced excessive bold text.")
    return "\n\n".join(out_blocks)


def _limit_bold(text: str, keep: int, max_words: int) -> str:
    count = 0

    def _sub(m):
        nonlocal count
        inner = m.group(1)
        if len(inner.split()) > max_words:
            return inner
        count += 1
        return m.group(0) if count <= keep else inner

    return re.sub(r"\*\*([^*]+)\*\*", _sub, text)


def split_long_paragraphs(text: str, fixes: list[str], max_words: int = 70, max_sentences: int = 4) -> str:
    """Walls of text become paragraphs of at most three sentences."""
    out = []
    changed = False
    for block in blocks_of(text):
        if block_kind(block) != "text":
            out.append(block)
            continue
        sentences = split_sentences(block)
        if len(sentences) <= max_sentences and len(words(block)) <= max_words:
            out.append(block)
            continue
        chunk: list[str] = []
        for sentence in sentences:
            if chunk and (len(chunk) >= 3 or len(words(" ".join(chunk + [sentence]))) > max_words):
                out.append(" ".join(chunk))
                chunk = []
            chunk.append(sentence)
        if chunk:
            out.append(" ".join(chunk))
        changed = True
    if changed:
        _note(fixes, "Split long paragraphs into shorter ones.")
    return "\n\n".join(out)


# --- removing sentences, items and rows ---------------------------------------

def remove_sentences(text: str, predicate: Callable[[str], bool],
                     kinds: Optional[set[str]] = None) -> tuple[str, list[str]]:
    """Drop every sentence (in prose), item (in a list) or row (in a table)
    for which `predicate(plain_text)` is true. Returns (text, removed).
    `kinds` limits which block kinds are inspected (default: all)."""
    removed: list[str] = []
    out_blocks = []
    for block in blocks_of(text):
        kind = block_kind(block)
        if kinds is not None and kind not in kinds:
            out_blocks.append(block)
            continue
        if kind == "table":
            rows = block.split("\n")
            keep = rows[:2]
            for row in rows[2:]:
                plain = _plain(row.replace("|", " "))
                if predicate(plain):
                    removed.append(plain)
                else:
                    keep.append(row)
            if len(keep) > 2:
                out_blocks.append("\n".join(keep))
            continue
        if kind == "list":
            items = block.split("\n")
            keep = []
            for item in items:
                plain = _plain(_LIST.sub("", item))
                if predicate(plain):
                    removed.append(plain)
                else:
                    keep.append(item)
            if keep:
                out_blocks.append("\n".join(keep))
            continue
        if kind == "heading":
            out_blocks.append(block)
            continue
        prefix = ""
        body = block
        if kind == "quote":
            prefix = "> "
            body = re.sub(r"^\s*>\s?", "", block, flags=re.M)
        kept = []
        for sentence in split_sentences(body):
            if predicate(_plain(sentence)):
                removed.append(_plain(sentence))
            else:
                kept.append(sentence)
        if kept:
            out_blocks.append(prefix + " ".join(kept))
    # A heading left with nothing under it goes too.
    cleaned: list[str] = []
    for i, block in enumerate(out_blocks):
        if block_kind(block) == "heading" and (i + 1 >= len(out_blocks) or block_kind(out_blocks[i + 1]) == "heading"):
            continue
        cleaned.append(block)
    return "\n\n".join(cleaned), removed


def _plain(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = re.sub(r"\*\*|__|(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)|^#+\s*|^\s*>\s?", "", text)
    return " ".join(text.split())


def fix_links(text: str, fixes: list[str], allowed_urls: Iterable[str],
              allowed_contacts: Iterable[str]) -> tuple[str, list[str]]:
    """Placeholder links and untraceable URLs/contacts.

    A link to an allowed URL stays. A link to anything else keeps its
    words and loses the link. A placeholder ("[link]", "your-website.com",
    "+91 XXXXX") or a contact detail the brief never gave takes its whole
    sentence with it — a sentence built around a fake phone number has
    nothing left to say without it."""
    allowed_urls = [u for u in allowed_urls if u]
    contacts = {c.strip().lower() for c in allowed_contacts if c}
    contact_digits = {_phone_digits(c) for c in contacts if _phone_digits(c)}
    removed: list[str] = []

    def _link(m):
        label, url = m.group(1), m.group(2)
        if url_allowed(url, allowed_urls):
            return m.group(0)
        _note(fixes, "Removed links that were placeholders or not from a verified source.")
        if not url or url.startswith("#") or find_placeholders(url) or not re.match(r"^(?:https?://|www\.)", url):
            return _PLACEHOLDER_MARK + label      # the sentence existed to carry a fake link
        return label

    text = _MD_LINK.sub(_link, text)

    def _bad(sentence: str) -> bool:
        if find_placeholders(sentence):
            return True
        for url in _BARE_URL.findall(sentence):
            if not url_allowed(url.rstrip(".,;"), allowed_urls):
                return True
        for email in _EMAIL.findall(sentence):
            if email.lower() not in contacts:
                return True
        for phone in _PHONE.findall(sentence):
            if _is_phone(phone) and _phone_digits(phone) not in contact_digits:
                return True
        return False

    if _PLACEHOLDER_MARK in text or any(_bad(_plain(b)) for b in blocks_of(text)):
        text, dropped = remove_sentences(text, lambda s: _PLACEHOLDER_MARK in s or _bad(s))
        text = text.replace(_PLACEHOLDER_MARK, "")
        if dropped:
            removed.extend(d.replace(_PLACEHOLDER_MARK, "") for d in dropped)
            _note(fixes, f"Removed {len(dropped)} sentence(s) containing placeholder or unverified "
                         "links/contact details.")
    return text, removed


# ---------------------------------------------------------------------------
# The whole pass
# ---------------------------------------------------------------------------

def normalise_markdown(md: str, *, allowed_urls: Iterable[str] = (), allowed_contacts: Iterable[str] = (),
                       allow_headings: bool = True, limits: Optional[dict] = None) -> NormaliseResult:
    """Every mechanical fix, in the order that makes each one safe."""
    limits = limits if limits is not None else blog_config.limits()
    fixes: list[str] = []
    issues: list[tuple[str, str, str]] = []
    text = preclean(md, fixes)
    text = fix_lines(text, fixes, allow_headings=allow_headings)
    text = separate_blocks(text, fixes)

    blocks = []
    for block in blocks_of(text):
        if block_kind(block) == "table":
            block = normalise_table(block, fixes)
        else:
            block = fix_inline(block, fixes)
        if block.strip():
            blocks.append(block)
    text = "\n\n".join(blocks)

    text = normalise_units(text, fixes)
    text, dropped = replace_stock_phrases(text, fixes)
    text, removed = fix_links(text, fixes, allowed_urls, allowed_contacts)
    for sentence in removed:
        issues.append(("placeholder_removed", "info", f'Removed: "{sentence[:90]}"'))
    text = cap_bold(text, fixes, limits.get("bold_max_per_paragraph", 1))
    text = split_long_paragraphs(text, fixes, limits.get("paragraph_max_words", 70),
                                 limits.get("paragraph_max_sentences", 4))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return NormaliseResult(text=text, fixes=fixes, issues=issues)


# ---------------------------------------------------------------------------
# Leak detection on RENDERED text
# ---------------------------------------------------------------------------

_LEAKS = [
    ("bold markers", re.compile(r"\*\*")),
    ("heading marks", re.compile(r"(?:^|\s)#{2,6}\s")),
    ("table rule", re.compile(r"\|\s*:?-{3,}")),
    ("link markup", re.compile(r"\]\(")),
    ("underscore emphasis", re.compile(r"(?<!\w)__\w")),
    ("code fence", re.compile(r"```")),
    ("collapsed bullets", re.compile(r"(?:\s[-•]\s[A-Z][^.\n]{2,40}){3,}")),
]


def markup_leaks(rendered_text: str) -> list[str]:
    """Raw markdown visible in rendered text (what a reader would see)."""
    found = []
    for name, pattern in _LEAKS:
        if pattern.search(rendered_text or ""):
            found.append(name)
    return found
