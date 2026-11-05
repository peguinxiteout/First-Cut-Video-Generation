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

THE BUDGET. The lines follow a structure for the video's length (hook,
features, proof, call to action), then lines are dropped from the end or
added until the words are within the tolerance of the target (about 2.1
words a second): a script that does not fit its length would be sped up
until it cannot be followed.
"""
from __future__ import annotations

import logging
import random
import re
import zlib
from dataclasses import dataclass, replace
from typing import Optional

from src.agents.brochure import facts, strategy, topics
from src.agents.brochure.copywriter import ZONE_PHRASE, fill, slots
from src.agents.video import photo_tags
from src.agents.video._brochure import short_item
from src.agents.video.brief import pace
from src.agents.video.guard import VideoGuard
from src.agents.video.storyboard import hook_asset
from src.core.config import load_yaml_config
from src.data.schemas import PropertyInput, PropertyKnowledge
from src.data.video_schemas import RankedAsset, ScriptLine, VideoBrief, VideoScript

logger = logging.getLogger(__name__)

MAX_SELLING_POINT_WORDS = 10      # a longer selling point is a paragraph, not a line to say
MAX_CHIP_WORDS = 5
MAX_EXTRA_LINES = 3               # lines added to reach the word target, beyond the structure
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


@dataclass
class Result:
    script: VideoScript
    concept: dict


def _lower(item: str) -> str:
    return item if any(c.isupper() for c in item[1:]) else item.lower()


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


def _structure(duration_s: int) -> list[str]:
    table = load_yaml_config("video/script_templates.yaml")["structure"]
    return list(table[max(k for k in table if k <= duration_s)])


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
        self.values = {**slots(self.content, self.positioning), "cta": brief.cta_text}
        self.rng = random.Random(zlib.crc32(self.content.name.encode("utf-8")))
        self.notes: list[str] = []
        self.last_opener = ""         # how the previous line began: the next one begins differently

    def refuse(self, what: str, text: str, reason: str) -> None:
        self.notes.append(f"{what} dropped: '{text}' because {reason}.")

    def pick(self, frames: list[str], values: Optional[dict] = None, max_words: int = 0) -> str:
        """The first frame (from a point that depends on the property's name)
        that fills, fits and passes the guard; "" if none does. A frame that
        begins like the previous line's is only used when no other will do."""
        values = values or self.values
        frames = list(frames)
        start = self.rng.randrange(len(frames)) if frames else 0
        ordered = frames[start:] + frames[:start]
        fresh = [f for f in ordered if _opener(f) != self.last_opener]
        for frame in fresh + [f for f in ordered if f not in fresh]:
            text = fill(frame, values, self.content)
            text = _spoken(text) if text else text
            if not text or (max_words and len(text.split()) > max_words):
                continue
            reason = self.guard.why(text)
            if reason:
                self.refuse("Line", text, reason)
                continue
            self.last_opener = _opener(frame)
            return self.guard.keep(text)
        return ""

    # -- what can be said ---------------------------------------------------------

    @staticmethod
    def _amenity(text: str) -> bool:
        """Names something a picture would have to show (a utility or a service does not)."""
        return bool(photo_tags.needs(text) or (topics.match(text, topics.AMENITY_TOPICS) and not photo_tags.non_visual(text)))

    def features(self, photo_topics: set[str]) -> list[Feature]:
        out: list[Feature] = []
        for highlight in self.brief.highlights:
            reason = self.guard.why(_sentence(highlight), repeat=False)
            if reason:
                self.refuse("Highlight", highlight, reason)
                continue
            out.append(Feature("highlight", topics.match(highlight), text=_sentence(highlight),
                               chip=highlight if len(highlight.split()) <= MAX_CHIP_WORDS else "",
                               amenity=self._amenity(highlight), units=_units([highlight])))
        for point in self.selling_points():
            out.append(Feature("selling_point", topics.match(point), text=_sentence(point),
                               amenity=self._amenity(point), units=_units([point])))
        zones = [z for z in self.content.zones if ZONE_PHRASE.get(z.topic)]
        zones.sort(key=lambda z: z.topic not in photo_topics)       # zones we have a photograph of first
        for zone in zones:
            items = []
            for item in zone.items:
                short = short_item(item)
                reason = self.guard.why(short, repeat=False) if short else ""
                if reason:
                    self.refuse("Amenity", item, reason)
                elif short:
                    items.append(short)
            if items:
                units = _units(items[:MAX_ZONE_ITEMS])
                out.append(Feature("zone", zone.topic, item_list=facts.prose_list([_lower(u) for u in units[:2]]),
                                   chip=items[0], amenity=any(not photo_tags.non_visual(u) for u in units),
                                   units=units))
        return out

    def selling_points(self) -> list[str]:
        """The ledger's selling points short enough to say as they are."""
        points = []
        for point in self.knowledge.selling_points:
            text = facts.clean(point).rstrip(".")
            if 3 <= len(text.split()) <= MAX_SELLING_POINT_WORDS and not self.guard.why(_sentence(text)):
                points.append(text)
        return points

    def feature_line(self, feature: Feature) -> Optional[ScriptLine]:
        if feature.source == "zone":
            text = self.pick(self.tone["zone"], {**self.values, "item_list": feature.item_list})
        else:
            spoken = _spoken(feature.text)
            reason = self.guard.why(spoken)
            if reason:
                self.refuse("Feature", feature.text, reason)
                return None
            self.last_opener = ""
            text = self.guard.keep(spoken)
        return ScriptLine(text=text, role="feature", topic=feature.topic) if text else None


def _label(feature: Feature) -> str:
    return "'" + (feature.text or feature.item_list or feature.chip).rstrip(".") + "'"


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
        after = facts.prose_list([_lower(u) for u in fit.kept[:2]])
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
    tagged = any(a.tags for a in usable)              # read by a vision model: follow the photographs
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
                w.notes.append(f"Left out of the script: {_label(feature)}. No photograph shows it.")
        pending = backed + general                    # what a picture shows first, then the rest

    kept: list[tuple[ScriptLine, Optional[Feature]]] = []

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
                feature, trim = _said(feature, fit)
            line = w.feature_line(feature)
            if line:
                if fit is not None:
                    claimed.add(fit.photo.image_id)
                    line.photo, line.tags = fit.photo.image_id, fit.tags
                    if trim:
                        w.notes.append(trim)
                return line, feature
        return None

    for role in _structure(brief.duration_s):
        if role == "hook":
            text = (w.pick(w.tone["hook"][objective], max_words=max_hook)
                    or w.pick(["{short}."], max_words=max_hook) or brief.name)
            kept.append((ScriptLine(text=text, role="hook", topic="exterior"), None))
        elif role == "feature":
            built = next_feature()
            if built:
                kept.append(built)
        elif role == "proof":
            text = w.pick(w.tone["proof"])
            if text:
                kept.append((ScriptLine(text=text, role="proof"), None))
        elif role == "cta":
            text = w.pick(w.tone["cta"][objective]) or w.pick(["{cta}."]) or facts.sentence(brief.cta_text)
            kept.append((ScriptLine(text=text, role="cta"), None))

    # Fit the words to the length: drop from the end, or add before the call to action.
    target, tolerance = brief.target_words, w.templates["word_tolerance"]

    def words() -> int:
        return sum(line.words for line, _ in kept)

    photo_shots = brief.n_shots - 1                   # a line that needs a picture shown has a shot of its own

    def room() -> bool:
        return not tagged or sum(1 for line, _ in kept if line.role != "cta") < photo_shots

    while tagged and not room() and sum(1 for line, _ in kept if line.role != "cta") > photo_shots:
        index = next((i for i in range(len(kept) - 1, 0, -1) if kept[i][0].role in ("proof", "feature")), None)
        if index is None:
            break
        w.notes.append(f"Trimmed to the {photo_shots} photograph shots: '{kept[index][0].text}'.")
        kept.pop(index)
    while words() > target * (1 + tolerance):
        index = next((i for i in range(len(kept) - 1, 0, -1) if kept[i][0].role in ("proof", "feature")), None)
        if index is None:
            break
        w.notes.append(f"Trimmed to fit {brief.duration_s} s: '{kept[index][0].text}'.")
        kept.pop(index)
    added = 0
    while words() < target * (1 - tolerance) and added < MAX_EXTRA_LINES and room():
        built = next_feature()
        if built is None:
            break
        if words() + built[0].words > target * (1 + tolerance):
            continue                       # too long to fit: try a shorter one
        kept.insert(len(kept) - 1, built)
        added += 1
    if not target * (1 - tolerance) <= words() <= target * (1 + tolerance):
        w.notes.append(f"The script is {words()} words against a target of {target}: "
                       "add highlights or more property details for a fuller script.")

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
