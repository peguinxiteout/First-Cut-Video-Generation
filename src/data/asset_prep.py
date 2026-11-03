"""Normalise uploaded images before they are laid out (P2).

The design review traced several "unfinished" symptoms to the source files
rather than the layout:

* The master plan drawing filled only ~60% of its own image; the rest was
  white canvas baked into the file, so scaling the file made the drawing
  small no matter which layout was used. -> trim_uniform_border()
* Photos supplied as screenshots of an existing brochure carry its legal
  text and "ARTISTIC RENDITION" flattened into the pixels, which then
  appear stacked with ours. -> burned_in_text_warning()
* A plan saved with a transparent background can render as white lines on
  black. -> flatten_onto_white()
* An image can be too low-resolution for the size it is printed at, and
  nothing said so. -> print_dpi()

Trimming a uniform border removes blank canvas, not content, so it does
not conflict with this project's never-crop rule: nothing that carries
information is lost. Only drawings (plans, documents, diagrams) are
trimmed; photographs are never touched.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageChops

logger = logging.getLogger(__name__)

# Categories whose images are drawings on a blank ground.
DRAWING_CATEGORIES = {"floorplan", "map", "document", "infographic"}

A4_WIDTH_INCHES = 8.27          # the brochure page's printed width
MIN_PRINT_DPI = 200             # below this, a printed image looks soft
BORDER_TOLERANCE = 10           # per-channel tolerance when judging "uniform"
MIN_TRIM_FRACTION = 0.04        # ignore trims smaller than this (not worth rewriting the file)
# A brochure page is 8.27in wide. At 300 dpi (print standard) a full-width
# image needs ~2480px; anything larger only inflates the PDF. An exported
# file was 54 MB — 3.7x the original brochure — almost entirely source
# images embedded at full camera/render resolution.
MAX_PLACED_WIDTH_PX = 3508      # an A4 long edge at 300 dpi
JPEG_QUALITY = 88

# Text baked into a photo that indicates it came from an existing brochure.
_BURNED_IN_PATTERNS = re.compile(
    r"rera|registration no|up-rera|artistic (rendition|impression)|artist'?s impression|"
    r"indicative|not to scale|www\.[a-z0-9.-]+",
    re.IGNORECASE,
)


@dataclass
class AssetIssue:
    image_id: str
    kind: str          # trimmed | flattened | low_resolution | burned_in_text
    message: str


def flatten_onto_white(img: Image.Image) -> Image.Image:
    """Composite transparency onto white. A transparent plan drawn in dark
    ink is invisible (or inverted) on a dark ground otherwise."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])
        return canvas
    return img.convert("RGB")


def uniform_border_box(img: Image.Image) -> Optional[tuple[int, int, int, int]]:
    """Bounding box of the actual content, if the image has a uniform
    border (white canvas around a drawing). None when there isn't one."""
    rgb = img.convert("RGB")
    corner = rgb.getpixel((0, 0))
    background = Image.new("RGB", rgb.size, corner)
    diff = ImageChops.difference(rgb, background).convert("L")
    # Tolerance: near-white scans are not exactly 255,255,255.
    box = diff.point(lambda v: 255 if v > BORDER_TOLERANCE else 0).getbbox()
    if not box:
        return None
    width, height = rgb.size
    trimmed = (box[0] + (width - box[2]) + box[1] + (height - box[3]))
    if trimmed < MIN_TRIM_FRACTION * (width + height):
        return None
    if (box[2] - box[0]) < width * 0.2 or (box[3] - box[1]) < height * 0.2:
        return None  # suspiciously aggressive: leave the image alone
    return box


def print_dpi(pixel_width: int, page_width_fraction: float = 1.0) -> float:
    """Effective print resolution if the image is placed across
    `page_width_fraction` of the page width."""
    return pixel_width / max(0.01, A4_WIDTH_INCHES * page_width_fraction)


def burned_in_text_warning(image_id: str, visible_text: str, category: str) -> Optional[AssetIssue]:
    """Photographs carrying legal/disclaimer text in their pixels. Drawings
    legitimately contain text, so they're exempt."""
    if category in DRAWING_CATEGORIES or not visible_text:
        return None
    match = _BURNED_IN_PATTERNS.search(visible_text)
    if not match:
        return None
    return AssetIssue(image_id, "burned_in_text",
                      f'Photo has text baked into it ("{match.group(0)}"). It will appear alongside the '
                      f"brochure's own disclaimer — supply a clean original if you have one.")


def normalise_image(
    source: Path, dest_dir: Path, image_id: str, category: str, placement_fraction: float = 1.0,
    grade=None,
) -> tuple[Path, list[AssetIssue]]:
    """Returns the path to use (original, or a normalised copy) and any issues."""
    issues: list[AssetIssue] = []
    try:
        img = Image.open(source)
    except Exception as exc:  # noqa: BLE001 - a bad file must not stop the brochure
        logger.warning("Could not open %s: %s", source, exc)
        return source, issues

    changed = False
    flat = flatten_onto_white(img)
    if flat.mode != img.mode:
        changed = True
        issues.append(AssetIssue(image_id, "flattened", f"{source.name}: transparent background filled with white."))

    if category in DRAWING_CATEGORIES:
        box = uniform_border_box(flat)
        if box:
            before = flat.size
            flat = flat.crop(box)
            changed = True
            issues.append(AssetIssue(
                image_id, "trimmed",
                f"{source.name}: trimmed {before[0] - flat.size[0]}x{before[1] - flat.size[1]}px of blank "
                f"border, so the drawing itself fills the space."))

    # THE GRADE. One grade on every photograph in the book, so a set shot
    # by three people on three days reads as one set — and so a generated
    # impression sits with the photographs rather than beside them. Never
    # on a drawing: a filter on a floor plan is a change to what it
    # states. See src/data/grading.py, including why this is not a CSS
    # filter (it costs 45MB on a twenty-eight page book).
    if grade is not None and category not in DRAWING_CATEGORIES and not grade.is_identity:
        from src.data import grading

        graded = grading.apply(flat, grade)
        if graded is not flat:
            flat = graded
            changed = True

    dpi = print_dpi(flat.size[0], placement_fraction)
    if dpi < MIN_PRINT_DPI:
        issues.append(AssetIssue(
            image_id, "low_resolution",
            f"{source.name}: about {dpi:.0f} dpi at the size it is placed ({MIN_PRINT_DPI}+ recommended for "
            f"print) — it may look soft."))

    if flat.size[0] > MAX_PLACED_WIDTH_PX:
        ratio = MAX_PLACED_WIDTH_PX / flat.size[0]
        before = flat.size
        flat = flat.resize((MAX_PLACED_WIDTH_PX, max(1, int(flat.size[1] * ratio))), Image.LANCZOS)
        changed = True
        issues.append(AssetIssue(
            image_id, "downsampled",
            f"{source.name}: {before[0]}x{before[1]} -> {flat.size[0]}x{flat.size[1]}px "
            f"(300 dpi at full page width); keeps the PDF small without visible loss."))

    if not changed:
        return source, issues
    dest_dir.mkdir(parents=True, exist_ok=True)
    if category in DRAWING_CATEGORIES:
        if source.suffix.lower() in (".jpg", ".jpeg", ".webp"):
            # already lossy at the source: a high-quality JPEG loses nothing
            # more, and is a tenth of the size of a lossless copy
            out = dest_dir / f"{source.stem}_normalised.jpg"
            flat.save(out, "JPEG", quality=92, optimize=True, subsampling=0)
            return out, issues
        out = dest_dir / f"{source.stem}_normalised.png"   # line art stays lossless
        flat.save(out)
        return out, issues

    # Photography: keep whichever encoding is actually smaller. JPEG usually
    # wins by a wide margin on renders and photographs, but it loses on
    # high-frequency or flat-colour images — picking blindly can make a file
    # BIGGER, which is the opposite of the point.
    as_jpeg = dest_dir / f"{source.stem}_normalised.jpg"
    as_png = dest_dir / f"{source.stem}_normalised.png"
    flat.save(as_jpeg, quality=JPEG_QUALITY, optimize=True, progressive=True)
    flat.save(as_png, optimize=True)
    if as_jpeg.stat().st_size <= as_png.stat().st_size:
        as_png.unlink(missing_ok=True)
        return as_jpeg, issues
    as_jpeg.unlink(missing_ok=True)
    return as_png, issues


def trim_logo(source: Path, dest_dir: Path, image_id: str) -> Path:
    """A logo with its empty margin removed and its transparency KEPT, so it
    sits on any ground. A logo on an opaque ground is trimmed of a uniform
    border of that ground."""
    try:
        img = Image.open(source)
        img.load()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not open logo %s: %s", source, exc)
        return source
    rgba = img.convert("RGBA")
    box = rgba.split()[-1].getbbox()
    if box and box != (0, 0) + rgba.size:
        rgba = rgba.crop(box)
    else:
        opaque = uniform_border_box(rgba.convert("RGB"))
        if not opaque:
            return source
        rgba = rgba.crop(opaque)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{source.stem}_logo.png"
    rgba.save(out)
    return out
