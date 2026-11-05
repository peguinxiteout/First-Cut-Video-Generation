"""The Video tab's prompt box: a free-text prompt in, the video options it states out.

    read_prompt(prompt, property_dir, offline=False)  -> PromptBriefResult(brief, notes, source)
    prompt_overrides(result)                          -> only the options the prompt set

Rules always run (configs/video/prompt_brief.yaml): length and shape, the objective, tone, audience,
music mood, captions, a quoted call to action, the amenities asked for, and the rest as creative
direction. A model (the creative_director role) is then asked once, with no retry and a short
timeout; where it returns a valid value it overrides the rule's. Offline, with no key, a spent
allowance, a timeout or an answer that is not JSON, the rules alone give the brief and a note says why.
Length and shape are never asked of the model, and the voice, the opening picture, generated frames
and the grade are never set from a prompt.

A prompt adds no fact. A highlight is kept only when the property has it, in the property's own
words: its amenities, selling points and amenity groups, and the fact ledger when one is cached.
The tag vocabulary (photo_tags) helps match wording ("lap pool" is the property's "Swimming pool");
it never admits a highlight by itself. A highlight or call to action with a figure that is not one
of the facts' own is left out. What was left out or not used is in the notes, in plain words.

The call is made in the Streamlit process, so it runs under policy.model_call_policy(), which limits
the gateway for this call only and changes no setting.
"""
from __future__ import annotations

import contextvars
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Optional

from src.agents.brochure import catalog, property_intelligence
from src.agents.video import fact_ledger, guard, photo_tags, policy
from src.core.config import load_yaml_config
from src.core.exceptions import ModelOutputParseError
from src.data.documents import load_documents
from src.data.ingestion import load_property_input
from src.data.schemas import PropertyInput
from src.data.video_schemas import VIDEO_DURATIONS, VIDEO_OBJECTIVES, PromptBrief
from src.models.gateway import call_with_fallback

logger = logging.getLogger(__name__)

ROLE = "creative_director"
CONFIG_FILE = "video/prompt_brief.yaml"
SOURCE_MODEL, SOURCE_RULES = "model", "rules"
_STOP = {"the", "a", "an", "and", "of", "with", "for", "in", "on", "to", "at", "our", "its", "all", "some", "your",
         "their"}
_QUOTED = re.compile(r"[“\"]([^”\"]{4,120})[”\"]")
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_NOT_A_LENGTH = r"(?!\s+(?:from|away|walk|drive|to|by|of|floor|level)\b)"   # "2 minutes from the metro" is a distance
_CLAUSE = re.compile(r"\s+(?:for|with|in|at|on|to|by|using|from|that|which|while|so|but|then|as)\b", re.I)


@dataclass
class PromptBriefResult:
    brief: PromptBrief
    notes: list[str] = field(default_factory=list)       # what was cut, left out, snapped or not used
    source: str = SOURCE_RULES                           # "model" when the model's answer was read


@dataclass
class _Facts:
    """What the property has, in its own words, and the figures its facts contain."""
    amenities: list[str]
    sellings: list[str]
    figures: set[str]
    words: set[str]

    @property
    def things(self) -> list[str]:
        return [*self.amenities, *self.sellings]


def _config() -> dict:
    return load_yaml_config(CONFIG_FILE)


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _tokens(text: str) -> set[str]:
    """The content words of a text, plural endings dropped, to compare wording."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _norm(text).split() if w not in _STOP}


def _phrases(words) -> re.Pattern:
    """Any of the words or phrases, whole, longest first."""
    alternation = "|".join(re.escape(str(w)) for w in sorted(words, key=lambda w: -len(str(w))))
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.I)


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _trim(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip(" ,;:-")


def _words(items: list[str], limit: int = 4) -> str:
    return ", ".join(list(dict.fromkeys(i.lower() for i in items))[:limit])


# ---------------------------------------------------------------------------
# The property's facts
# ---------------------------------------------------------------------------

def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        item = re.sub(r"\s+", " ", str(item or "")).strip()
        if item and _norm(item) not in seen:
            seen.add(_norm(item))
            out.append(item)
    return out


def _facts_of(prop: PropertyInput, property_dir: Path) -> _Facts:
    """The property's own wording: property.json, and the ledger when one is cached (it can add what a
    document states). Never asks a model."""
    try:
        documents_text, _ = load_documents(property_dir)
        knowledge = fact_ledger.cached(prop, documents_text)
    except Exception:  # noqa: BLE001 - a ledger that cannot be looked up is no reason to fail the prompt
        knowledge = None
    knowledge = knowledge or property_intelligence.ledger_from_form(prop)
    from_ledger = re.split(r"[,;\n]", knowledge.facts.get("amenities", ""))
    amenities = _dedupe([*prop.amenities, *(i for g in prop.amenity_groups for i in g.items), *from_ledger])
    sellings = _dedupe([*prop.usps, *knowledge.selling_points])
    text = " ".join([prop.model_dump_json(), *knowledge.facts.values(), *knowledge.selling_points,
                     *knowledge.specifications, *knowledge.legal_notes, knowledge.rera_number]).replace(",", "")
    return _Facts(amenities, sellings, guard.figures(text), guard.number_words(text))


def _match(item: str, facts: _Facts) -> Optional[str]:
    """The property's own wording for something a prompt asks to feature, or None when it has no such thing.
    A vocabulary amenity ("lap pool") needs one of the property's things that names the same amenities;
    anything else needs all its words in one of them."""
    wanted = photo_tags.needs(item)
    if wanted:
        return next((t for t in facts.things if wanted <= photo_tags.needs(t)), None)
    tokens = _tokens(item)
    return next((t for t in facts.things if tokens and tokens <= _tokens(t)), None) if tokens else None


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _length_matches(text: str, cfg: dict) -> list[tuple[int, int, float]]:
    """(start, end, seconds) of every length the prompt names, in order. A length keyword ("teaser") counts
    only when no length is named outright."""
    found: list[tuple[int, int, float]] = []
    for m in re.finditer(rf"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*(seconds?|secs?|s)\b{_NOT_A_LENGTH}", text, re.I):
        found.append((m.start(), m.end(), float(m.group(1))))
    for m in re.finditer(rf"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*(minutes?|mins?)\b{_NOT_A_LENGTH}", text, re.I):
        found.append((m.start(), m.end(), float(m.group(1)) * 60))
    for m in re.finditer(rf"\b(?:a|one)\s*-?\s*minute\b{_NOT_A_LENGTH}", text, re.I):
        found.append((m.start(), m.end(), 60.0))
    for m in re.finditer(r"\bhalf(?:\s+a)?[\s-]+minute\b", text, re.I):
        found.append((m.start(), m.end(), 30.0))
    spoken = {k.replace("-", " "): v for k, v in (cfg["length"]["words"] or {}).items()}
    for m in re.finditer(rf"\b({'|'.join(re.escape(k) for k in sorted(spoken, key=len, reverse=True))})[\s-]*"
                         rf"(?:seconds?|secs?)\b{_NOT_A_LENGTH}", text.replace("-", " "), re.I):
        found.append((m.start(), m.end(), float(spoken[m.group(1).lower()])))
    minutes = cfg["length"].get("minute_words") or {}
    if minutes:
        for m in re.finditer(rf"\b({'|'.join(re.escape(k) for k in minutes)})[\s-]*(?:minutes?|mins?)\b{_NOT_A_LENGTH}",
                             text, re.I):
            found.append((m.start(), m.end(), float(minutes[m.group(1).lower()]) * 60))
    if not found:
        for word, seconds in (cfg["length"]["keywords"] or {}).items():
            m = _phrases([word]).search(text)
            if m:
                found.append((m.start(), m.end(), float(seconds)))
    return sorted(found)


def _length(text: str, cfg: dict, notes: list[str]) -> Optional[int]:
    """Seconds named in the prompt, snapped to an offered length."""
    found = _length_matches(text, cfg)
    if not found:
        return None
    named = list(dict.fromkeys(s for _, _, s in found))
    seconds = found[0][2]
    if len(named) > 1:
        notes.append(f"Several lengths are named ({', '.join(_number(s) + ' s' for s in named)}); using the first.")
    best = min(VIDEO_DURATIONS, key=lambda d: (abs(d - seconds), d))             # a tie goes to the shorter
    if best != seconds:
        notes.append(f"Length {_number(seconds)} s is not offered; using {best} s "
                     f"(the lengths are {', '.join(map(str, VIDEO_DURATIONS[:-1]))} and {VIDEO_DURATIONS[-1]}).")
    return best


def _first_of(families: dict[str, list], text: str) -> list[tuple[str, int]]:
    """(family, position of its first word) for each family the text names, earliest first."""
    out = []
    for name, words in families.items():
        m = _phrases(words).search(text) if words else None
        if m:
            out.append((name, m.start()))
    return sorted(out, key=lambda x: x[1])


def _shape(text: str, cfg: dict, notes: list[str]) -> Optional[str]:
    """The shape named first. One pass over every shape's words, longest first, so "YouTube Shorts" is a
    vertical video and not a YouTube one."""
    owner = {str(w).lower(): shape for shape, words in cfg["shape"].items() for w in words or []}
    named = list(dict.fromkeys(owner[m.group(0).lower()] for m in _phrases(owner).finditer(text)))
    if len(named) > 1:
        notes.append(f"Several shapes are named ({', '.join(named)}); using {named[0]}, the first.")
    return named[0] if named else None


def _objective(text: str, cfg: dict) -> Optional[str]:
    """The objective with the most mentions; a tie goes to the one named first."""
    counts = {name: len(_phrases(words).findall(text)) for name, words in cfg["objective"].items() if words}
    first = dict(_first_of(cfg["objective"], text))
    ranked = sorted((n for n, c in counts.items() if c), key=lambda n: (-counts[n], first[n]))
    return ranked[0] if ranked else None


def _tone(text: str, cfg: dict) -> Optional[str]:
    tones = _tones()
    families = {name: [name, *((cfg["tone_synonyms"] or {}).get(name) or [])] for name in tones}
    counts = {name: len(_phrases(words).findall(text)) for name, words in families.items()}
    first = dict(_first_of(families, text))
    ranked = sorted((n for n, c in counts.items() if c), key=lambda n: (-counts[n], first[n]))
    return ranked[0] if ranked else None


def _tones() -> list[str]:
    return list(load_yaml_config("video/script_templates.yaml")["tones"])


def _audience(text: str, cfg: dict) -> Optional[str]:
    by_lower = {str(a).lower(): str(a) for a in cfg["audience"]}
    m = _phrases(by_lower).search(text)
    return by_lower[m.group(0).lower()] if m else None


def _music_match(text: str, cfg: dict) -> Optional[tuple[str, int, int]]:
    """(the mood words, where they start, where the noun ends) of a music request ("warm acoustic music")."""
    moods = {str(m).lower() for m in cfg["music_moods"]}
    for m in _phrases(cfg["music_nouns"]).finditer(text):
        before = list(re.finditer(r"[\w'-]+|&", text[:m.start()].lower()))[-4:]
        picked: list[str] = []
        start = m.start()
        for token in reversed(before):
            word = token.group(0)
            if word in moods:
                picked.insert(0, word)
                start = token.start()
            elif word in ("and", "&") and picked:
                continue
            else:
                break
        if picked:
            return " ".join(picked), start, m.end()
    return None


def _music(text: str, cfg: dict) -> Optional[str]:
    match = _music_match(text, cfg)
    return match[0] if match else None


def _captions(text: str, cfg: dict) -> Optional[bool]:
    if _phrases(cfg["captions_off"]).search(text):
        return False
    return True if _phrases(cfg["captions_on"]).search(text) else None


def _cta(text: str, cfg: dict) -> Optional[str]:
    """A call to action the prompt words itself: a quoted line after a cue, or the line after "CTA:"."""
    cues = "|".join(re.escape(c) for c in sorted(cfg["cta_cues"], key=len, reverse=True))
    quoted = re.search(rf"(?<!\w)(?:{cues})\s*(?:is|are|should be|will be|:|=|-)?\s*[“\"]([^”\"]{{3,80}})[”\"]",
                       text, re.I)
    plain = re.search(r"(?<!\w)(?:cta|call[- ]to[- ]action)\s*[:=]\s*([^\n\"“.;]{3,80})", text, re.I)
    m = quoted or plain
    return m.group(1).strip(" ,;:-") if m else None


def _cue_items(text: str, cfg: dict) -> list[str]:
    """The things listed after a highlight cue ("highlight the pool, gym and clubhouse")."""
    cues = "|".join(re.escape(c) for c in sorted(cfg["highlight_cues"], key=len, reverse=True))
    not_used = _phrases([w for family in cfg["not_used"] for w in family["words"]])
    generic = {str(g).lower() for g in cfg["generic_items"]}
    items: list[str] = []
    for m in re.finditer(rf"(?<!\w)(?:{cues})(?!\w)[:\s-]*(?:(?:the|our|its|these|such as|like|of|a|an)\s+)*([^.!?\n;]+)",
                         text, re.I):
        listed = _CLAUSE.split(" " + m.group(1), maxsplit=1)[0]            # the list ends where a clause begins
        for part in re.split(r",|&|/|\band\b|\bplus\b", listed):
            item = re.sub(r"^(?:the|a|an|our|its|all|some|their)\s+", "", part.strip(" \"'“”.-"), flags=re.I)
            if item and len(item.split()) <= 5 and item.lower() not in generic and not not_used.search(item):
                items.append(item)
    return items


def _vocabulary_items(text: str, cfg: dict, facts: _Facts) -> list[str]:
    """The property's things named by a vocabulary word anywhere in the prompt (one per amenity), without the
    words that describe a picture rather than ask for something ("lush green trees")."""
    rest = " ".join(s for s in _SENTENCES.split(text) if not _cue_items(s, cfg))      # a cue's items are checked as asked
    said = photo_tags.needs(_phrases(cfg["weak_words"]).sub(" ", rest))
    out: list[str] = []
    for tag in sorted(said):
        thing = next((t for t in facts.things if tag in photo_tags.needs(t)), None)
        if thing and thing not in out:
            out.append(thing)
    return [t for t in facts.things if t in out]                      # the property's own order


def _consumed(text: str, cfg: dict) -> list[tuple[int, int]]:
    """Where the rules have already read something: a length, a shape, the audience, the objective's words,
    the captions and the music request."""
    spans = [(a, b) for a, b, _ in _length_matches(text, cfg)]
    spans += [m.span() for word in (cfg["length"]["keywords"] or {}) for m in _phrases([word]).finditer(text)]
    owner = {str(w).lower(): shape for shape, words in cfg["shape"].items() for w in words or []}
    spans += [m.span() for m in _phrases(owner).finditer(text)]
    spans += [m.span() for m in _phrases([str(a) for a in cfg["audience"]]).finditer(text)]
    spans += [m.span() for words in cfg["objective"].values() if words for m in _phrases(words).finditer(text)]
    spans += [m.span() for m in _phrases([*cfg["captions_off"], *cfg["captions_on"]]).finditer(text)]
    music = _music_match(text, cfg)
    if music:
        spans.append((music[1], music[2]))
    return spans


def _residual(sentence: str, filler: set[str]) -> str:
    """What a sentence still says once what the rules read is blanked out: each clause trimmed of the
    little words at its ends, and a clause with nothing left dropped."""
    def bare(word: str) -> str:
        return word.strip(".,;:!?\"'()-\u201c\u201d").lower()

    clauses = []
    for clause in re.split(r"[,;:]", sentence):
        words = clause.split()
        while words and bare(words[0]) in filler:
            words.pop(0)
        while words and bare(words[-1]) in filler:
            words.pop()
        if words:
            clauses.append(" ".join(words))
    return ", ".join(clauses)


def _direction(text: str, cfg: dict, cta: Optional[str], facts: _Facts, notes: list[str]) -> Optional[str]:
    """What is left of the prompt once the rules have read what they can: sentences about a call to action,
    quoted lines, highlight lists and what the video cannot do are taken out, and from the others the words
    the rules read (length, shape, audience, objective, captions, music) and the little words around them.
    Nothing left: None. A sentence with a figure that is not a fact (a price, a discount) is not passed on
    either: direction steers the music, but it is not the place for a claim. The length the prompt names does
    not count as a figure."""
    not_used = _phrases([w for family in cfg["not_used"] for w in family["words"]])
    cues = _phrases(cfg["cta_cues"])
    filler = {"", *(str(w).lower() for w in cfg["filler"])}
    masked = text
    for start, end in _consumed(text, cfg):
        masked = masked[:start] + " " * (end - start) + masked[end:]
    kept, claims = [], []
    for sentence in (s.strip() for s in _SENTENCES.split(_QUOTED.sub(" ", masked))):
        if (not sentence or not_used.search(sentence) or _cue_items(sentence, cfg)
                or (cta and (cues.search(sentence) or cta.lower() in sentence.lower()))):
            continue
        if guard.unknown_figure(sentence, facts.figures, facts.words):
            claims.append(sentence.rstrip(".!? "))
            continue
        left = _residual(sentence, filler).rstrip(".!? ")
        if left:
            kept.append(left[0].upper() + left[1:] + ".")
    if claims:
        notes.append(f"Not used as creative direction (a figure that is not in this property's facts): "
                     f"'{_trim('; '.join(re.sub(chr(32) + '+', chr(32), c) for c in claims), 120)}'.")
    return _trim(" ".join(kept), cfg["limits"]["creative_direction"]) or None


def _not_used(text: str, cfg: dict, cta: Optional[str]) -> list[str]:
    notes = []
    for family in cfg["not_used"]:
        found = [m.group(0) for m in _phrases(family["words"]).finditer(text)]
        if found:
            notes.append(f"{family['note']} (found: {_words(found)}).")
    lines = [q for q in _QUOTED.findall(text) if len(q.split()) >= 2 and _norm(q) != _norm(cta or "")]
    if lines:
        notes.append(f"Quoted lines are not printed ({len(lines)}): the video's words are written from the property's "
                     "facts; only a call to action is taken from a prompt.")
    return notes


def _rules(text: str, cfg: dict, facts: _Facts, notes: list[str]) -> dict[str, Any]:
    cta = _cta(text, cfg)
    values: dict[str, Any] = {
        "duration_s": _length(text, cfg, notes), "aspect": _shape(text, cfg, notes), "objective": _objective(text, cfg),
        "audience": _audience(text, cfg), "tone": _tone(text, cfg), "music_mood": _music(text, cfg),
        "burn_captions": _captions(text, cfg), "cta": cta, "creative_direction": _direction(text, cfg, cta, facts, notes),
        "highlights": _dedupe([*_cue_items(text, cfg), *_vocabulary_items(text, cfg, facts)])}
    notes.extend(_not_used(text, cfg, cta))
    return values


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def _model_prompt(text: str, prop: PropertyInput, facts: _Facts, cfg: dict) -> str:
    things = "\n".join(f"- {_trim(t, cfg['limits']['highlight'])}" for t in facts.things[:cfg["max_things"]])
    return Template(cfg["prompt"]).substitute(
        name=prop.name.strip(), category=catalog.normalise_category(prop.category) or "unspecified",
        things=things or "(none listed)", objectives=", ".join(VIDEO_OBJECTIVES), tones=", ".join(_tones()),
        prompt=text)


def _ask(prompt: str, timeout_s: float) -> Any:
    """One call, no retry, given up on after `timeout_s`. It runs in a thread (the provider has no timeout of
    its own) that carries this context, so the gateway's call policy applies there."""
    def call():
        with policy.model_call_policy():
            return call_with_fallback(ROLE, "generate_json", prompt)

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(contextvars.copy_context().run, call).result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False)


def _why_not(exc: Exception, timeout_s: float) -> str:
    """The failure of a model call in a few plain words (never the provider's own text)."""
    text = str(exc).lower()
    if isinstance(exc, TimeoutError):
        return f"it did not answer within {_number(timeout_s)} s"
    if "blocked by the run's call policy" in text:
        return "the run's policy does not allow it"
    if isinstance(exc, ModelOutputParseError):
        return "its answer was not JSON"
    if any(w in text for w in ("429", "quota", "resource_exhausted", "rate limit", "rate-limit")):
        return "its allowance is spent"
    if any(w in text for w in ("is not set", "api key", "401", "403", "unauthenticated", "permission")):
        return "no valid model key is set"
    return f"it could not be asked ({type(exc).__name__})"


def _text(value: Any, limit: int) -> Optional[str]:
    return _trim(value, limit) or None if isinstance(value, str) else None


def _model_values(raw: Any, cfg: dict, notes: list[str]) -> dict[str, Any]:
    """The values of a model's answer that are usable; each unusable one is left out and named."""
    if not isinstance(raw, dict):
        notes.append("The model's answer was not a JSON object; the rules were used.")
        return {}
    limits, values = cfg["limits"], {}

    def refuse(key: str, why: str = "was not usable") -> None:
        notes.append(f"The model's {key.replace('_', ' ')} {why}; left out.")

    objective = raw.get("objective")
    if objective is not None:
        clean = str(objective).strip().lower().replace(" ", "_").replace("-", "_")
        values["objective"] = clean if clean in VIDEO_OBJECTIVES else None
        if values["objective"] is None:
            refuse("objective", f"('{objective}') is not one of {', '.join(VIDEO_OBJECTIVES)}")
    tone = raw.get("tone")
    if tone is not None:
        clean = str(tone).strip().lower()
        values["tone"] = clean if clean in _tones() else None
        if values["tone"] is None:
            refuse("tone", f"('{tone}') is not one of {', '.join(_tones())}")
    for key in ("audience", "cta", "music_mood", "creative_direction"):
        if raw.get(key) is not None:
            values[key] = _text(raw[key], limits[key])
            if values[key] is None:
                refuse(key)
    highlights = raw.get("highlights")
    if highlights is not None:
        if isinstance(highlights, list) and all(isinstance(h, str) for h in highlights):
            values["highlights"] = _dedupe([_trim(h, limits["highlight"]) for h in highlights])[:cfg["max_highlights"] * 2]
        else:
            refuse("highlights")
    if raw.get("burn_captions") is not None:
        if isinstance(raw["burn_captions"], bool):
            values["burn_captions"] = raw["burn_captions"]
        else:
            refuse("captions choice")
    pronunciations = raw.get("pronunciations")
    if pronunciations is not None:
        if isinstance(pronunciations, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                    for k, v in pronunciations.items()):
            values["pronunciations"] = {k.strip(): v.strip() for k, v in pronunciations.items()
                                        if k.strip() and v.strip()} or None
        else:
            refuse("pronunciations")
    return {k: v for k, v in values.items() if v not in (None, [], {})}


# ---------------------------------------------------------------------------
# Facts check
# ---------------------------------------------------------------------------

def _as_written(text: str, figure: str) -> str:
    """The figure as the text writes it ("00000", not "0")."""
    return next((t for t in re.findall(r"\d[\d,]*(?:\.\d+)?", text) if figure in guard.figures(t)), figure)


def _check_highlights(items: list[str], facts: _Facts, cfg: dict, notes: list[str]) -> list[str]:
    kept: list[str] = []
    for item in items:
        figure = guard.unknown_figure(item, facts.figures, facts.words)
        if figure:
            notes.append(f"'{item}' has a figure ({_as_written(item, figure)}) that is not in this property's "
                         "facts; left out.")
            continue
        thing = _match(item, facts)
        if thing is None:
            notes.append(f"'{item}' is not in this property's details; left out.")
        elif thing not in kept:
            kept.append(thing)
    if len(kept) > cfg["max_highlights"]:
        notes.append(f"Only the first {cfg['max_highlights']} highlights are used.")
    return kept[:cfg["max_highlights"]]


def _check_cta(cta: Optional[str], facts: _Facts, notes: list[str]) -> Optional[str]:
    figure = guard.unknown_figure(cta, facts.figures, facts.words) if cta else ""
    if figure:
        notes.append(f"The call to action has a figure ({_as_written(cta, figure)}) that is not in this property's "
                     "facts; left out.")
        return None
    return cta


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def read_prompt(prompt: str, property_dir: Path, offline: bool = False, *,
                timeout_s: Optional[float] = None) -> PromptBriefResult:
    """The options a prompt states for the property at `property_dir`, checked against its facts.
    Raises IngestionError when the folder has no readable property.json."""
    cfg = _config()
    timeout_s = float(cfg["model_timeout_s"] if timeout_s is None else timeout_s)
    notes: list[str] = []
    text = (prompt or "").strip()
    if len(text) > cfg["max_chars"]:
        text = text[:cfg["max_chars"]].rstrip()
        notes.append(f"The prompt is longer than {cfg['max_chars']:,} characters; only the first "
                     f"{cfg['max_chars']:,} were read.")
    prop = load_property_input(Path(property_dir))
    facts = _facts_of(prop, Path(property_dir))

    values = _rules(text, cfg, facts, notes)
    source = SOURCE_RULES
    blocker = ("the prompt is empty" if not text else "an offline run" if offline
               else "" if policy.role_listed(ROLE) else f"the run's policy does not list the '{ROLE}' model")
    if not blocker:
        try:
            answer = _ask(_model_prompt(text, prop, facts, cfg), timeout_s)
        except Exception as exc:  # noqa: BLE001 - whatever stopped the model, the rules still give a brief
            blocker = _why_not(exc, timeout_s)
            logger.warning("The model was not used for the prompt (%s: %s)", type(exc).__name__, blocker)
        else:
            source = SOURCE_MODEL
            values.update(_model_values(answer, cfg, notes))
    if source == SOURCE_RULES:
        notes.append("The prompt is empty." if not text else
                     f"The model was not used ({blocker}); the prompt was read with built-in rules.")

    values["highlights"] = _check_highlights(values.get("highlights") or [], facts, cfg, notes) or None
    values["cta"] = _check_cta(values.get("cta"), facts, notes)
    values["pronunciations"] = values.get("pronunciations") or None
    return PromptBriefResult(PromptBrief(**values), list(dict.fromkeys(notes)), source)


def max_prompt_chars() -> int:
    """The longest prompt that is read (a longer one is cut, with a note)."""
    return int(_config()["max_chars"])


def length_of(prompt: str) -> Optional[int]:
    """The length a prompt names, snapped to an offered one (rules only, no property needed)."""
    return _length((prompt or "").strip()[:_config()["max_chars"]], _config(), [])


def shape_of(prompt: str) -> Optional[str]:
    """The shape a prompt names (rules only, no property needed)."""
    return _shape((prompt or "").strip()[:_config()["max_chars"]], _config(), [])


def prompt_overrides(result: PromptBriefResult) -> dict[str, Any]:
    """Only the options the prompt set, as the overrides of a video (laid over video_input.json)."""
    return result.brief.model_dump(exclude_none=True)
