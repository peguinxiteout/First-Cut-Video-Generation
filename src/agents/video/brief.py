"""Node 1, brief: what is being made, for whom, at what size.

Reads the property folder (the same `data/properties/<id>/` the brochure
reads) and the optional `video_input.json` beside `property.json`, lays the
command-line options over it, and resolves every default into one
VideoBrief, so no later step has to work a default out again. A property
with no `video_input.json` gets a 15 second 9:16 video.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from src.agents.brochure import catalog, palettes
from src.core.config import load_yaml_config
from src.core.exceptions import IngestionError
from src.data.documents import load_documents
from src.data.ingestion import discover_images, load_property_input
from src.data.schemas import ImageAsset, PropertyInput
from src.data.video_schemas import BrandKit, VideoBrief, VideoInput

logger = logging.getLogger(__name__)

VIDEO_INPUT_FILE = "video_input.json"


@dataclass
class Brief:
    property_input: PropertyInput
    images: list[ImageAsset]
    documents_text: str
    video_input: VideoInput
    brief: VideoBrief
    brand_kit: BrandKit
    notes: list[str] = field(default_factory=list)


def load_video_input(property_dir: Path, overrides: dict | None = None) -> VideoInput:
    """`video_input.json` (all defaults when there is none) with the
    `overrides` laid over it. An option left as None is not given; an empty
    string or list is, and clears what the file says. Raises IngestionError,
    in words, when the file or an option is not valid."""
    path = property_dir / VIDEO_INPUT_FILE
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise IngestionError(f"{VIDEO_INPUT_FILE} in {property_dir} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise IngestionError(f"{VIDEO_INPUT_FILE} in {property_dir} must be a JSON object")
    data = {**data, **{k: v for k, v in (overrides or {}).items() if v is not None}}
    try:
        return VideoInput.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise IngestionError(f"The video options are not valid: {problems}") from exc


def _tone(requested: str, category: str, notes: list[str]) -> str:
    known = load_yaml_config("video/script_templates.yaml")["tones"]
    tone = (requested or "").strip().lower()
    if tone in known:
        return tone
    default = load_yaml_config("video/pacing.yaml")["category_tone"].get(category, "warm")
    if tone:
        notes.append(f"Tone '{requested}' is not one of {', '.join(known)}; using '{default}'.")
    return default


def pace(category: str, tone: str) -> tuple[str, float]:
    """The pace of the cut (its name, and seconds per photograph): the tone's
    if it sets one, else the category's."""
    pacing = load_yaml_config("video/pacing.yaml")
    name = pacing["tone_pace"].get(tone) or pacing["category_pace"].get(category, "steady")
    return name, float(pacing["paces"][name])


def _shots(duration_s: int, category: str, tone: str) -> int:
    """Shots in the whole video, the closing card included."""
    end_card_s = load_yaml_config("video/pacing.yaml")["end_card_s"]
    photo_shots = round((duration_s - end_card_s) / pace(category, tone)[1])
    return max(2, photo_shots) + 1


def resolve_brief(prop: PropertyInput, video_input: VideoInput, *, draft: bool, notes: list[str]) -> VideoBrief:
    pacing = load_yaml_config("video/pacing.yaml")
    category = catalog.normalise_category(prop.category) or "mid"
    quality = "draft" if draft else "final"
    width, height = pacing["sizes"][quality][video_input.aspect]
    tone = _tone(video_input.tone, category, notes)
    templates = load_yaml_config("video/script_templates.yaml")
    words_per_second = load_yaml_config("video/speech.yaml")["words_per_second"]
    highlights = list(dict.fromkeys(h.strip() for h in video_input.highlights if h and h.strip()))
    fields = video_input.model_dump()
    fields.update(
        highlights=highlights, tone=tone,
        music_mood=video_input.music_mood.strip() or pacing["music_mood"].get(category, ""),
        cta=video_input.cta.strip(), hero_image=video_input.hero_image.strip())
    return VideoBrief(
        **fields, property_id=prop.property_id, name=prop.name.strip(), developer=(prop.developer or "").strip(),
        category=category, width=width, height=height, fps=pacing["fps"][quality], quality=quality,
        target_words=int((video_input.duration_s - pacing["end_card_s"]) * words_per_second + 0.5),
        n_shots=_shots(video_input.duration_s, category, tone),
        palette_name=palettes.for_category(category).name,
        cta_text=video_input.cta.strip() or (prop.cta or "").strip() or templates["cta_defaults"][video_input.objective])


def brand_kit(prop: PropertyInput, images: list[ImageAsset], category: str) -> BrandKit:
    palette = palettes.with_brand(palettes.for_category(category), prop.brand_color)
    logo = next((i for i in images if i.role == "logo"), None)
    contact = prop.contact
    return BrandKit(
        name=prop.name.strip(), logo_path=logo.path if logo else None, palette_name=palette.name,
        palette=palette.tokens(), accent=palette.accent, rera_number=(prop.rera_number or "").strip(),
        phone=((contact.phone if contact else "") or "").strip(),
        website=((contact.website if contact else "") or "").strip())


def run(property_dir: Path, overrides: dict | None = None, *, draft: bool = False) -> Brief:
    prop = load_property_input(property_dir)
    images = discover_images(property_dir)
    documents_text, document_notes = load_documents(property_dir)
    video_input = load_video_input(property_dir, overrides)
    notes = list(document_notes)
    brief = resolve_brief(prop, video_input, draft=draft, notes=notes)
    return Brief(prop, images, documents_text, video_input, brief, brand_kit(prop, images, brief.category), notes)
