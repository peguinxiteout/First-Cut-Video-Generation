"""Node 4, script: what the narrator says and what the screen prints.

The offline script. Every sentence is a frame from
configs/video/script_templates.yaml filled ONLY from the property's own
facts (the brochure's content model and slots), or a highlight the person
typed, or a short selling point of the property. Nothing is composed by a
model here.

THE GUARD. Each narration line and each piece of on-screen text goes through
VideoGuard. A line it refuses is dropped or replaced by the next safe frame,
and the reason is written to the script's notes, so a highlight that was left
out is never a mystery.

THE PHOTOGRAPHS. When the pictures were read by a vision model (they carry tags), the
script follows them. A feature that names amenities (a pool, a jacuzzi, a barbecue) is
split into its units (the items of a zone, the parts of "X and Y"), a photograph is
chosen that shows the most of them, and the line says ONLY those: "Fitness Studio and
spa" over a gym becomes "Fitness Studio". The photograph is set aside for the line
(`ScriptLine.photo`) and every trim is written to the notes. A feature none of whose
units any photograph shows is left out, with a note; a unit no photograph can show (a
utility, security) and a general line (the design, the location) need no picture.
Without tags (offline) the script is chosen as before.

THE BUDGET. About one line a photograph shot: the hook, then as many body lines as the video has
photograph shots less the hook's, added until the words reach the target (about 2.1 words a second;
a script that does not fit its length would have to be sped up, a short one leaves silence). The
body comes from the features above, then from general lines about the property, written from its
own facts only (agents/video/slots.py: its homes by bedrooms, its scale, its location, possession,
price and developer) and from selling points cut down by cutting, never by rewording. A line has at
least three words: a frame that fills to "Brigade." is not a line. Lines are ordered so that no
topic follows itself; the hook stays first and the call to action last. Every line, whatever its
source, passes VideoGuard.
"""
from __future__ import annotations

import logging
import math
import random
import re
import zlib
from dataclasses import dataclass, replace
from typing import Optional

from src.agents.brochure import facts, strategy, topics
from src.agents.brochure.copywriter import ZONE_PHRASE, fill, slots
from src.agents.video import photo_tags
from src.agents.video import slots as video_slots
from src.agents.video._brochure import short_item
from src.agents.video.brief import pace
from src.agents.video.guard import VideoGuard
from src.agents.video.storyboard import hook_asset, photo_shot_count
from src.core.config import load_yaml_config
from src.data.schemas import PropertyInput, PropertyKnowledge
from src.data.video_schemas import RankedAsset, ScriptLine, VideoBrief, VideoScript

logger = logging.getLogger(__name__)

MAX_SELLING_POINT_WORDS = 10      # a longer selling point is a paragraph, not a line to say
MAX_CHIP_WORDS = 5
MAX_ZONE_ITEMS = 6                # items of an amenity zone looked at when choosing what to say


@dataclass
class Feature:
    """Something worth a line: a highlight, a selling point, or an amenity zone."""
    source: str                   # highlight | selling_point | zone
    topic: str                    # what a photograph of it shows
    text: str = ""                # the line, for a highlight or selling point
    item_list: str = ""           # for a zone: "swimming pool and splash pool"
    chip: str = ""                # the short on-screen form
    amenity: bool = False         # it names something a picture would have to show
    units: tuple[str, ...] = ()   # what it names, one each: the items of a zone, the parts of "X and Y"
    origin: str = ""              # the fact it comes from, for ScriptLine.source


@dataclass
class Result:
    script: VideoScript
    concept: dict


def _plain(text: str) -> str:
    """Lower case words only, to find a unit's words among the facts' words."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


_SPLIT = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*|\s*/\s*")


def _parts(text: str) -> Optional[tuple[str, ...]]:
    """The amenities of "X, Y and Z", when the text is nothing but such a list (each part names an
    amenity or something no picture can show); else None."""
    parts = [p.strip() for p in _SPLIT.split(text.strip().rstrip(".")) if p and p.strip()]
    if len(parts) >= 2 and all(photo_tags.needs(p) or photo_tags.non_visual(p) for p in parts):
        return tuple(dict.fromkeys(parts))
    return None


def _units(items: list[str]) -> tuple[str, ...]:
    """The units of a feature's texts: a text that is a list is split, any other stays whole."""
    units: list[str] = []
    for item in items:
        for unit in _parts(item) or (item,):
            if unit not in units:
                units.append(unit)
    return tuple(units)


@dataclass
class Fit:
    """A feature as one photograph lets it be said."""
    photo: RankedAsset
    kept: list[str]               # the units said
    dropped: list[str]            # the units this photograph does not show
    tags: list[str]               # the photograph's tags that show the kept units


def _fit(feature: Feature, asset: RankedAsset) -> Optional[Fit]:
    """The units of a feature that one photograph shows, or None when it shows none of them. A unit is
    shown when EVERY amenity it names is a tag of the picture: the line says the unit, and the gate
    reads the line the same way."""
    kept, dropped, shown = [], [], []
    for unit in feature.units:
        if photo_tags.non_visual(unit):
            kept.append(unit)                                    # nothing to show: it is always fine to say
            continue
        needed = photo_tags.needs(unit)
        found = sorted(needed) if needed and needed <= set(asset.tags) else ([] if needed else photo_tags.shows(unit, asset.tags))
        if found:
            kept.append(unit)
            shown += [t for t in found if t not in shown]
        else:
            dropped.append(unit)
    return Fit(asset, kept, dropped, shown) if shown else None


def best_fit(feature: Feature, pictures: list[RankedAsset]) -> Optional[Fit]:
    """The picture that lets a feature say the most. Of pictures that let it say the same, the one
    with fewer tags: a picture that shows many things is kept for the lines that need them. Then
    the better picture."""
    best: Optional[Fit] = None
    for asset in pictures:
        found = _fit(feature, asset)
        if found and (best is None or (len(found.kept), -len(asset.tags)) > (len(best.kept), -len(best.photo.tags))):
            best = found
    return best


def _spoken(text: str) -> str:
    """Narration is read aloud: a slash ("tennis/pickleball") is said as "and".
    On-screen text keeps it."""
    return re.sub(r"\s*/\s*", " and ", text)


def _opener(frame: str) -> str:
    """How a frame begins: its words up to the first slot, or the slot itself."""
    frame = re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", frame)
    return frame.split("{")[0].strip() or "{" + frame.split("}")[0].lstrip("{") + "}"


def _sentence(text: str) -> str:
    return facts.sentence(text) if text.rstrip()[-1:] not in "?!" else facts.clean(text)


def _remainder(feature: Feature, fit: Fit) -> Optional[Feature]:
    """The units one photograph did not show, as a feature of their own: each unit then gets a picture
    that shows it (a court and a fitness area are two lines over two photographs, not one trimmed line)."""
    if not fit.dropped:
        return None
    units = tuple(fit.dropped)
    if feature.source == "zone":
        return replace(feature, units=units, item_list=facts.prose_list(list(units[:2])), chip=units[0])
    short = len(" ".join(units).split()) <= MAX_CHIP_WORDS
    return replace(feature, units=units, text=_sentence(facts.prose_list(list(units))),
                   chip=units[0] if feature.chip and short else "")


def _spread(body: list) -> list:
    """The body lines ordered so that no topic follows itself where another line can go between. The
    order is kept otherwise."""
    out, rest = [], list(body)
    while rest:
        previous = out[-1][0].topic if out else ""
        index = next((i for i, (line, _) in enumerate(rest) if not (line.topic and line.topic == previous)), 0)
        out.append(rest.pop(index))
    return out


# slot -> (topic, the fact it comes from, the picture it wants). A date or a price fits any picture.
PROOF_SLOTS = {"config": ("homes", "unit types and location", "exterior"), "possession": ("possession", "possession", "any"),
               "price": ("price", "price", "any"), "developer": ("developer", "developer", "exterior")}


class _Writer:
    def __init__(self, prop: PropertyInput, brief: VideoBrief, knowledge: PropertyKnowledge):
        self.brief = brief
        self.knowledge = knowledge
        self.templates = load_yaml_config("video/script_templates.yaml")
        self.pacing = load_yaml_config("video/pacing.yaml")
        self.tone = self.templates["tones"][brief.tone]
        self.positioning = strategy.for_category(brief.category)
        self.content = facts.build(prop, brief.category)
        self.guard = VideoGuard(prop, knowledge, self.content, self.positioning)
        self.values = {**slots(self.content, self.positioning), **video_slots.video_slots(prop), "cta": brief.cta_text}
        self.min_words = self.templates["line_budget"]["min_words"]
        self.max_claim_words = self.templates["line_budget"]["max_claim_words"]
        self.rng = random.Random(zlib.crc32(self.content.name.encode("utf-8")))
        self.notes: list[str] = []
        self.last_opener = ""         # how the previous line began: the next one begins differently
        self.used: set[str] = set()   # frames already used in this video: another is chosen while one fits
        named = [*(prop.amenities or []), *brief.highlights, *knowledge.selling_points,
                 *(item for zone in self.content.zones for item in zone.items)]
        self.wording = " " + " ".join(_plain(text) for text in named) + " "     # what the property itself calls things

    def refuse(self, what: str, text: str, reason: str) -> None:
        self.notes.append(f"{what} dropped: '{text}' because {reason}.")

    def pick(self, frames: list[str], values: Optional[dict] = None, max_words: int = 0, min_words: int = 0,
             acres_ok: bool = False) -> str:
        """The first frame (from a point that depends on the property's name)
        that fills, fits and passes the guard; "" if none does. A frame not yet used in this video comes
        before one that was, and a frame that begins like the previous line's is only used when no other
        will do. A text under `min_words` words is not a line. An area (sq ft, acres) is never spoken;
        `acres_ok` lets the scale line say its acres."""
        values = values or self.values
        frames = list(frames)
        start = self.rng.randrange(len(frames)) if frames else 0
        ordered = frames[start:] + frames[:start]
        for frame in sorted(ordered, key=lambda f: (f in self.used, _opener(f) == self.last_opener)):
            text = fill(frame, values, self.content)
            text = _spoken(text) if text else text
            if not text or (max_words and len(text.split()) > max_words) or len(text.split()) < min_words:
                continue
            if video_slots.has_area(text, acres_ok):
                self.refuse("Line", text, "it gives an area, which is not spoken")
                continue
            reason = self.guard.why(text)
            if reason:
                self.refuse("Line", text, reason)
                continue
            self.last_opener = _opener(frame)
            self.used.add(frame)
            return self.guard.keep(text)
        return ""

    def without_area(self, what: str, text: str) -> str:
        """The text without the clauses that give an area (sq ft, acres), which a narrator never says; ""
        when nothing is left. The note says what was cut."""
        if not video_slots.has_area(text):
            return text
        cut = video_slots.without_area(text)
        if cut:
            self.wording += _plain(cut) + " "                     # the property's own words, with a clause cut out
            self.notes.append(f"{what} shortened: '{text}' to '{cut}' because it gave an area, which is not spoken.")
        else:
            self.refuse(what, text, "it gives an area, which is not spoken")
        return cut

    def named(self, unit: str) -> bool:
        """Whether the property's own facts use these words (an amenity, a highlight, a selling point): a line
        says what the property calls a thing, never a name from a photograph's tags."""
        return f" {_plain(unit)} " in self.wording

    # -- what can be said ---------------------------------------------------------

    @staticmethod
    def _amenity(text: str) -> bool:
        """Names something a picture would have to show (a utility or a service does not)."""
        return bool(photo_tags.needs(text) or (topics.match(text, topics.AMENITY_TOPICS) and not photo_tags.non_visual(text)))

    def features(self, photo_topics: set[str]) -> list[Feature]:
        out: list[Feature] = []
        for given in self.brief.highlights:
            highlight = self.without_area("Highlight", given)
            reason = self.guard.why(_sentence(highlight), repeat=False) if highlight else "it is empty"
            if reason:
                if highlight:
                    self.refuse("Highlight", highlight, reason)
                continue
            out.append(Feature("highlight", topics.match(highlight), text=_sentence(highlight),
                               chip=highlight if len(highlight.split()) <= MAX_CHIP_WORDS else "",
                               amenity=self._amenity(highlight), units=_units([highlight]),
                               origin=f"highlight: {given}"))
        for point, origin in self.selling_points():
            out.append(Feature("selling_point", topics.match(point), text=_sentence(point),
                               amenity=self._amenity(point), units=_units([point]), origin=f"selling point: {origin}"))
        zones = [z for z in self.content.zones if ZONE_PHRASE.get(z.topic)]
        zones.sort(key=lambda z: z.topic not in photo_topics)       # zones we have a photograph of first
        for zone in zones:
            items = []
            for item in zone.items:
                short = short_item(item)
                reason = self.guard.why(short, repeat=False) if short else ""
                if short and video_slots.has_area(short):
                    reason = "it gives an area, which is not spoken"
                if reason:
                    self.refuse("Amenity", item, reason)
                elif short:
                    items.append(short)
            if items:
                units = _units(items[:MAX_ZONE_ITEMS])
                out.append(Feature("zone", zone.topic, item_list=facts.prose_list(list(units[:2])),
                                   chip=items[0], amenity=any(not photo_tags.non_visual(u) for u in units),
                                   units=units, origin="amenities: " + ", ".join(items[:3])))
        return out

    def selling_points(self) -> list[tuple[str, str]]:
        """(the line, the selling point it comes from) for each ledger selling point that can be said: cut down
        by cutting only (slots.claim), at most `max_claim_words` words, and passing the guard."""
        points = []
        for point in self.knowledge.selling_points:
            text = video_slots.claim(self.without_area("Selling point", point), self.max_claim_words, self.min_words)
            if text and not self.guard.why(_sentence(text)):
                self.wording += _plain(text) + " "                  # the point's own words, with a parenthetical cut out
                points.append((text, facts.clean(point)))
        return points

    # -- general lines about the property, from its own facts -------------------------------

    def homes_line(self, interior: bool) -> Optional[ScriptLine]:
        text = self.pick(self.tone["general"]["homes"], min_words=self.min_words)
        if text and interior:                                          # an interior picture is left: say so first
            lead = self.pick(self.templates["connective"])
            text = f"{lead} {text}" if lead else text
        scene = "interior" if interior else "exterior"          # a home's rooms, else the buildings
        return ScriptLine(text=text, role="proof", topic="homes", source="unit types", scene=scene) if text else None

    def general_line(self, key: str, topic: str, source: str, scene: str = "any") -> Optional[ScriptLine]:
        text = self.pick(self.tone["general"][key], min_words=self.min_words, acres_ok=key == "scale")
        return ScriptLine(text=text, role="proof", topic=topic, source=source, scene=scene) if text else None

    def proof_lines(self, said_homes) -> list:
        """One thunk for each proof frame, so each can be a line of its own. `said_homes()` tells whether the
        homes line is already in the script (the configuration frame then says nothing new)."""
        def thunk(frame: str):
            def make() -> Optional[ScriptLine]:
                slot = next((k for k in re.findall(r"\{(\w+)\}", frame) if k in PROOF_SLOTS), "")
                if slot == "config" and said_homes():                   # the homes line has said it already
                    return None
                text = self.pick([frame], min_words=self.min_words)
                topic, source, scene = PROOF_SLOTS.get(slot, ("", "property facts", "any"))
                return ScriptLine(text=text, role="proof", topic=topic, source=source, scene=scene) if text else None
            return make
        return [thunk(frame) for frame in self.tone["proof"]]

    def feature_line(self, feature: Feature) -> Optional[ScriptLine]:
        for unit in feature.units:
            if not self.named(unit):
                self.refuse("Feature", unit, "the property's facts do not call anything by that name")
                return None
        if feature.source == "zone":
            text = self.pick(self.tone["zone"], {**self.values, "item_list": feature.item_list}, min_words=self.min_words)
        else:
            spoken = _spoken(feature.text)
            reason = self.guard.why(spoken)
            if reason:
                self.refuse("Feature", feature.text, reason)
                return None
            self.last_opener = ""
            text = self.guard.keep(spoken)
        scene = "amenity" if feature.amenity or feature.source == "zone" else "any"
        return ScriptLine(text=text, role="feature", topic=feature.topic, source=feature.origin, scene=scene) if text else None


def _label(feature: Feature) -> str:
    return "'" + (feature.text or feature.item_list or feature.chip).rstrip(".") + "'"


def _weak_photos(units, usable: list[RankedAsset]) -> list[str]:
    """The photographs the model tagged with an amenity of these units and whose caption does not say it
    (see photo_tags.evidence): in the frame, perhaps, but not clearly enough to show a viewer."""
    wanted = {name for unit in units for name in photo_tags.needs(unit)}
    return [a.filename for a in usable if wanted & set(a.weak_tags)]


def _weak_pairs(units, usable: list[RankedAsset]) -> list[str]:
    """"amenity (file)" for each amenity of these units that a photograph is tagged with and its caption does not say."""
    out: list[str] = []
    for unit in units:
        for name in sorted(photo_tags.needs(unit)):
            out += [f"{name} ({a.filename})" for a in usable if name in a.weak_tags and f"{name} ({a.filename})" not in out]
    return out


def _not_clear(weak: list[str]) -> str:
    if len(weak) == 1:
        return f"the only photograph tagged with it ({weak[0]}) does not show it clearly"
    return f"the photographs tagged with it ({', '.join(weak)}) do not show it clearly"


def _left_out(feature: Feature, usable: list[RankedAsset]) -> str:
    """Why an amenity feature has no line: no photograph shows it, or only a photograph tagged with it
    whose caption does not say it."""
    weak = _weak_photos(feature.units, usable)
    if not weak:
        return f"Left out of the script: {_label(feature)}. No photograph shows it."
    return f"{_label(feature)[1:-1]} left out: {_not_clear(weak)}."


def check(script: VideoScript, max_hook_words: int) -> list[str]:
    """The script's own sanity checks: problems, in words, or []."""
    if not script.lines:
        return ["the script has no lines"]
    problems = []
    if script.lines[0].role != "hook":
        problems.append("the first line is not the hook")
    if script.lines[0].words > max_hook_words:
        problems.append(f"the hook is {script.lines[0].words} words; the limit is {max_hook_words}")
    if script.lines[-1].role != "cta":
        problems.append("the last line is not the call to action")
    return problems


def _said(feature: Feature, fit: Fit) -> tuple[Feature, Optional[str]]:
    """The feature rewritten to the units the picture shows, and the note about what was trimmed (or None)."""
    if not fit.dropped:
        return feature, None
    if feature.source == "zone":
        before = feature.item_list
        shown = [u for u in fit.kept if not photo_tags.non_visual(u)] or fit.kept
        after = facts.prose_list(list(fit.kept[:2]))
        revised = replace(feature, item_list=after, chip=shown[0], units=tuple(fit.kept))
    else:
        before = feature.text
        after = _sentence(facts.prose_list(fit.kept))
        short = ", ".join(fit.kept) if len(" ".join(fit.kept).split()) <= MAX_CHIP_WORDS else ""
        revised = replace(feature, text=after, chip=short if feature.chip else "", units=tuple(fit.kept))
    if before.rstrip(".").lower() == after.rstrip(".").lower():
        return revised, None                                     # only units that would not have been said were dropped
    return revised, (f"Trimmed '{before.rstrip('.')}' to '{after.rstrip('.')}': "
                     f"no photograph shows {', '.join(d.lower() for d in fit.dropped)}.")


def run(prop: PropertyInput, brief: VideoBrief, knowledge: PropertyKnowledge,
        assets: Optional[list[RankedAsset]] = None) -> Result:
    w = _Writer(prop, brief, knowledge)
    max_hook = w.pacing["max_hook_words"]
    usable = [a for a in (assets or []) if a.usable]
    tagged = any(a.read_by_vision for a in usable)    # read by a vision model: follow the photographs
    pending = w.features({a.topic for a in usable})
    objective = brief.objective

    claimed: set[str] = set()                         # pictures already set aside for a line
    if tagged:
        hook = hook_asset(usable, brief.hero_image, [])
        if hook:
            claimed.add(hook.image_id)                # the picture the video opens on
        backed, general = [], []
        free = [a for a in usable if a.image_id not in claimed]
        for feature in pending:
            if not feature.amenity:
                general.append(feature)
            elif best_fit(feature, free):
                backed.append(feature)
            else:
                w.notes.append(_left_out(feature, usable))
        pending = backed + general                    # what a picture shows first, then the rest

    def next_feature() -> Optional[tuple[ScriptLine, Feature]]:
        while pending:
            feature = pending.pop(0)
            fit, trim = None, None
            if tagged and feature.amenity:
                fit = best_fit(feature, [a for a in usable if a.image_id not in claimed])
                if fit is None:
                    w.notes.append(f"Left out of the script: {_label(feature)}. "
                                   "The pictures that show it are used by other lines.")
                    continue
                before = feature
                feature, trim = _said(feature, fit)
                weak = _weak_pairs(fit.dropped, usable) if trim else []
                if weak:
                    trim = f"{trim} Tagged with it but not shown clearly: {'; '.join(weak)}."
                rest = _remainder(before, fit)
                if rest is not None and best_fit(rest, [a for a in usable if a.image_id not in claimed | {fit.photo.image_id}]):
                    pending.insert(0, rest)                      # what this picture does not show gets a picture of its own
            line = w.feature_line(feature)
            if line:
                if fit is not None:
                    claimed.add(fit.photo.image_id)
                    line.photo, line.tags = fit.photo.image_id, fit.tags
                    if trim:
                        w.notes.append(trim)
                return line, feature
        return None

    # The hook and the call to action first, so the body knows how many words are left for it.
    hook_text = (w.pick(w.tone["hook"][objective], max_words=max_hook) or w.pick(["{short}."], max_words=max_hook)
                 or brief.name)
    hook = ScriptLine(text=hook_text, role="hook", topic="exterior", scene="exterior")
    cta_text = w.pick(w.tone["cta"][objective]) or w.pick(["{cta}."]) or facts.sentence(brief.cta_text)
    cta = ScriptLine(text=cta_text, role="cta", scene="any")

    target, tolerance = brief.target_words, w.templates["word_tolerance"]
    lower, upper = math.floor(target * (1 - tolerance)), math.ceil(target * (1 + tolerance))     # whole words
    shots = photo_shot_count(brief, w.pacing)                      # the hook and the body: about a line each
    interior = tagged and video_slots.interior_photographs(usable)
    body: list[tuple[ScriptLine, Optional[Feature]]] = []

    def words() -> int:
        return hook.words + cta.words + sum(line.words for line, _ in body)

    extras = [lambda: w.homes_line(interior), lambda: w.general_line("scale", "scale", "project size", "view"),
              *w.proof_lines(lambda: any(line.source == "unit types" for line, _ in body))]
    extras.append(lambda: None if w.values.get("place") and any(w.values["place"] in line.text for line, _ in body)
                  else w.general_line("location", "location", "location", "view"))       # the place is already said

    def next_body() -> Optional[tuple[ScriptLine, Optional[Feature]]]:
        built = next_feature()
        while built is None and extras:
            line = extras.pop(0)()
            built = (line, None) if line else None
        return built

    room = shots + w.templates["line_budget"]["extra_lines"]       # a few short lines may share a shot
    while 1 + len(body) < room and words() < lower:
        built = next_body()
        if built is None:
            break
        if words() + built[0].words > upper:
            continue                                               # too long to fit: try the next
        body.append(built)
    while words() > upper and body:
        w.notes.append(f"Trimmed to fit {brief.duration_s} s: '{body[-1][0].text}'.")
        body.pop()
    if not lower <= words() <= upper:
        w.notes.append(f"The script is {words()} words against a target of {target}: "
                       "add highlights or more property details for a fuller script.")
    kept = [(hook, None), *_spread(body), (cta, None)]

    lines = [line for line, _ in kept]
    script = VideoScript(lines=lines, word_count=words(), mode="template")

    # What the screen prints.
    def screen(text: str, what: str) -> bool:
        reason = w.guard.why(text, repeat=False)
        if reason:
            w.refuse(what, text, reason)
        return not reason

    price = w.values.get("price", "")
    title = [brief.name] + ([f"From {price}"] if price else [])
    script.on_screen = [t for t in title if screen(t, "On-screen text")] or [brief.name]
    chips = [(f.chip, i) for i, (_, f) in enumerate(kept) if f and f.chip and screen(f.chip, "Chip")][:2]
    script.chips, script.chip_lines = [c for c, _ in chips], [i for _, i in chips]
    script.cta = brief.cta_text if screen(brief.cta_text, "Call to action") else w.templates["cta_defaults"][objective]
    rera = knowledge.rera_number or (prop.rera_number or "").strip()
    contact = prop.contact
    end_card = ([f"RERA: {rera}"] if rera else []) + [
        (contact.phone or "").strip() if contact else "", (contact.website or "").strip() if contact else ""]
    script.end_card = [t for t in end_card if t and screen(t, "End card text")]
    if not rera:
        w.notes.append("No RERA number is given for this property, so the closing card shows none.")
    script.notes = w.notes
    for problem in check(script, max_hook):
        script.notes.append(f"Check: {problem}.")

    pace_name, shot_s = pace(brief.category, brief.tone)
    concept = {"pace": pace_name, "shot_s": shot_s, "tone": brief.tone, "music_mood": brief.music_mood,
               "voice": brief.voice or "default", "target_words": target, "words": script.word_count,
               "structure": [line.role for line in lines]}
    return Result(script, concept)
