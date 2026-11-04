"""The number guard for what a video says and shows.

Everything spoken or drawn is checked here before it is used. It is the
brochure's Guard (copywriter.Guard) with the parts a video needs:

    * the facts are the property's own data AND its ledger (what a document
      or the model added), not only the form
    * a figure has to be one of the facts' own figures, not just a run of
      digits that happens to occur somewhere in them ("2" is not a fact
      because "2026" is)
    * number words count: a narrator saying "twenty" is a figure, and it is
      read aloud where a digit is only read
    * text with a damaged character is refused: it would be spoken as noise
      and drawn as a box

`why()` says which rule refused a line, so the script can record the reason
instead of dropping it silently."""
from __future__ import annotations

import re

from src.agents.brochure.copywriter import DESCRIPTIVE, Guard

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000, "dozen": 12,
}
# "zero-waste" and "zero-emission" are not quantities; "zero brokerage" and "twenty-five" are.
_WORD = re.compile(r"\b(" + "|".join(sorted((w for w in _WORDS if w != "zero"), key=len, reverse=True))
                   + r"|zero(?!-))\b", re.I)


def figures(text: str) -> set[str]:
    """Every figure in a text, in one spelling: no commas, no trailing zeros."""
    out = set()
    for token in _NUMBER.findall(text.replace(",", "")):
        value = float(token)
        out.add(str(int(value)) if value.is_integer() else repr(value))
    return out


def number_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


class VideoGuard(Guard):
    def __init__(self, prop, knowledge, content, positioning):
        super().__init__(prop, content, positioning)
        ledger = [*knowledge.facts.values(), *knowledge.selling_points, *knowledge.specifications,
                  *knowledge.legal_notes, knowledge.rera_number]
        self.facts = f"{self.facts} {' '.join(ledger)}".replace(",", "")
        self.known_figures = figures(self.facts)
        self.known_words = number_words(self.facts)

    def unknown_figure(self, text: str) -> str:
        """The first figure in the text that is not a fact, or ""."""
        for token in sorted(figures(text) - self.known_figures):
            return token
        for word in sorted(number_words(text) - self.known_words):
            if str(_WORDS[word]) not in self.known_figures:
                return word
        return ""

    def numbers_ok(self, text: str) -> bool:
        return not self.unknown_figure(text)

    def why(self, text: str, *, repeat: bool = True) -> str:
        """The rule that refuses this text, in words, or "" when it passes.
        `repeat=False` skips the check for a line already used, which an
        overlay (a name, a call to action) may repeat from the narration."""
        if not text or not text.strip():
            return "it is empty"
        if "�" in text:
            return "it contains a damaged character (the source text has a broken accent)"
        if DESCRIPTIVE.search(text):
            return "it describes the picture instead of selling the place"
        unknown = self.unknown_figure(text)
        if unknown:
            return f"'{unknown}' is not a figure in the property's facts"
        if not self.register_ok(text):
            return f"it uses a word the {self.positioning.label} register avoids"
        if repeat and self.key(text) in self.seen:
            return "it repeats a line already used"
        return ""

    def ok(self, text: str) -> bool:
        return not self.why(text)
