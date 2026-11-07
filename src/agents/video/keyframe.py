"""Keyframes for an image-to-video model: a photograph of any shape made into
the model's input in the video's shape (9:16 now), without changing what the
photograph shows.

An image-to-video model answers in the shape of the picture it is given, so a
1.89:1 photograph sent as it is comes back as a wide clip, and a wide clip in a
tall video is a letterbox. Three ways to give the model a tall picture:

    crop     a window of the video's shape around the subject (focal.py), enlarged
             to the model's input size. Real pixels only; the rest of the photograph
             is lost, and the enlargement is reported.
    extend   the whole photograph in the middle, untouched, and the missing top and
             bottom (or sides) filled by an outpaint model. The middle band is checked
             against what was sent and the original pixels are put back over it, so the
             building is the photograph's own; the filled areas are generated and the
             keyframe says so (`generated`, `band`).
    travel   two real windows of one wide photograph, a start and an end, for a model
             that takes a first and a last frame: the camera travels across the real
             picture instead of cutting most of it away.

`recommend` picks one per photograph from its shape, how much of its detail a crop
keeps (the same measure focal.py places windows with), and how far the crop is
enlarged. Thresholds are in configs/video/keyframe.yaml.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageOps

from src.agents.brochure import focal as focal_points
from src.core.config import load_yaml_config
from src.models.video_gen.base import VideoGenerationError
from src.models.video_gen.kenburns import crop_box

logger = logging.getLogger(__name__)

CONFIG_FILE = "video/keyframe.yaml"
MODES = ("crop", "extend", "travel")
JPEG_QUALITY = 95

# outpaint(picture, {"expand_top": px, "expand_bottom": px, "expand_left": px, "expand_right": px})
# -> the expanded picture. models/video_gen/fal.py has the hosted one; tests pass a fake.
Outpaint = Callable[[Image.Image, dict], Image.Image]


@dataclass
class Recommendation:
    mode: str
    reasons: list[str]
    aspect: float                    # the photograph's width over height
    crop_keep: float                 # share of the photograph's area a crop keeps
    crop_detail: float               # share of its detail a crop keeps
    crop_enlargement: float          # how much a crop's pixels are enlarged to the model's input
    subject_fits: bool


@dataclass
class ModelKeyframe:
    mode: str
    path: Path                       # the first frame sent to the model
    width: int
    height: int
    enlargement: float               # how much the photograph's pixels were enlarged
    focal: str = "50% 50%"
    end_path: Optional[Path] = None  # travel: the last frame
    generated: bool = False          # extend: holds areas made by a model
    band: Optional[tuple[int, int, int, int]] = None    # where the photograph's own pixels are (left, top, right, bottom)
    windows: list[tuple[int, int, int, int]] = field(default_factory=list)   # the source boxes used
    band_check: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)


def _cfg() -> dict:
    return load_yaml_config(CONFIG_FILE)


def target_size(aspect: str = "9:16") -> tuple[int, int]:
    sizes = _cfg()["input_size"]
    if aspect not in sizes:
        raise VideoGenerationError(f"no model input size for aspect {aspect!r}; add it to {CONFIG_FILE}")
    w, h = sizes[aspect]
    return int(w), int(h)


def _open(path: Path) -> Image.Image:
    try:
        with Image.open(path) as handle:
            return ImageOps.exif_transpose(handle).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - any unreadable picture is the same answer
        raise VideoGenerationError(f"cannot read {Path(path).name}: {exc}") from exc


def _save(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=JPEG_QUALITY, subsampling=0)
    return path


# ---------------------------------------------------------------------------
# How much a crop keeps
# ---------------------------------------------------------------------------

def _profile(path: Path, wide: bool) -> Optional[np.ndarray]:
    """Detail along the axis a crop cuts: columns for a wide photograph, rows for a tall one."""
    profiles = focal_points._energy(str(path))
    if profiles is None:
        return None
    return np.asarray(profiles[1] if wide else profiles[0], dtype=np.float64)


def _window_share(profile: np.ndarray, start: float, length: float) -> float:
    """Share of the profile inside [start, start + length), both as shares of its length."""
    total = float(profile.sum())
    if total <= 0:
        return length
    n = len(profile)
    a, b = int(round(start * n)), int(round((start + length) * n))
    return float(profile[max(0, a):min(n, b)].sum()) / total


def _subject_span(profile: Optional[np.ndarray], share: float) -> tuple[float, float]:
    """(start, end) of the stretch holding the middle `share` of the detail, as shares of the length."""
    if profile is None or float(profile.sum()) <= 0:
        return 0.0, 1.0
    cum = np.cumsum(profile) / float(profile.sum())
    tail = (1.0 - share) / 2
    n = len(profile)
    a = int(np.searchsorted(cum, tail))
    b = int(np.searchsorted(cum, 1.0 - tail)) + 1
    return a / n, min(1.0, b / n)


def recommend(path: Path, aspect: str = "9:16", topic: str = "") -> Recommendation:
    cfg = _cfg()["recommend"]
    tw, th = target_size(aspect)
    with Image.open(path) as handle:
        w, h = ImageOps.exif_transpose(handle).size
    img_aspect, slot_aspect = w / h, tw / th
    wide = img_aspect >= slot_aspect
    focal = focal_points.focal_point(str(path), img_aspect, slot_aspect, topic)
    left, top, right, bottom = crop_box(w, h, slot_aspect, focal)
    keep = (right - left) * (bottom - top) / (w * h)
    enlargement = max(tw / (right - left), th / (bottom - top))
    profile = _profile(path, wide)
    if profile is None:
        detail = keep
    elif wide:
        detail = _window_share(profile, left / w, (right - left) / w)
    else:
        detail = _window_share(profile, top / h, (bottom - top) / h)
    ratio = img_aspect / slot_aspect if wide else slot_aspect / img_aspect
    fits = detail >= cfg["crop_min_detail"]
    reasons = [f"{w}x{h} ({img_aspect:.2f}:1); a {aspect} crop keeps {keep:.0%} of the picture and "
               f"{detail:.0%} of its detail, enlarged {enlargement:.2f}x"]
    if abs(img_aspect - slot_aspect) / slot_aspect <= cfg["same_shape_tolerance"]:
        mode = "crop"
        reasons.append("already about the video's shape: nothing is lost")
    elif fits and enlargement <= cfg["crop_max_enlargement"]:
        mode = "crop"
        reasons.append("the subject fits one window")
    elif ratio >= cfg["travel_min_aspect_ratio"]:
        mode = "travel"
        reasons.append("the subject does not fit one window; a move across the real picture shows it")
    else:
        mode = "extend"
        reasons.append("the subject does not fit one window and the picture is not wide enough to travel "
                       "across; keep it whole and fill the rest")
    if enlargement > cfg["crop_max_enlargement"] and mode == "crop":
        reasons.append(f"a crop enlarges it {enlargement:.2f}x")
    if min(tw / w, th / h) > cfg["soft_enlargement"] or enlargement > cfg["soft_enlargement"]:
        reasons.append("the photograph is small for this size: any mode will look soft")
    return Recommendation(mode=mode, reasons=reasons, aspect=round(img_aspect, 3), crop_keep=round(keep, 3),
                          crop_detail=round(detail, 3), crop_enlargement=round(enlargement, 3), subject_fits=fits)


# ---------------------------------------------------------------------------
# The three modes
# ---------------------------------------------------------------------------

def crop(path: Path, dest: Path, aspect: str = "9:16", focal: Optional[str] = None, topic: str = "") -> ModelKeyframe:
    """A window of the video's shape around the subject, enlarged (LANCZOS) to the model's input size."""
    image = _open(path)
    tw, th = target_size(aspect)
    focal = focal or focal_points.focal_point(str(path), image.width / image.height, tw / th, topic)
    box = crop_box(image.width, image.height, tw / th, focal)
    enlargement = max(tw / (box[2] - box[0]), th / (box[3] - box[1]))
    _save(image.resize((tw, th), Image.LANCZOS, box=box), dest)
    notes = [f"crop {box} of {image.width}x{image.height}, enlarged {enlargement:.2f}x"]
    return ModelKeyframe(mode="crop", path=dest, width=tw, height=th, enlargement=round(enlargement, 3),
                         focal=focal, windows=[box], notes=notes)


def travel_windows(width: int, height: int, aspect: str = "9:16", profile: Optional[np.ndarray] = None
                   ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Start and end boxes of the video's shape inside a photograph: across the subject
    (the middle of its detail), at least `min_shift` and at most `max_shift` of a window apart."""
    cfg = _cfg()
    tw, th = target_size(aspect)
    wide = width / height >= tw / th
    length = width if wide else height
    win = round(height * tw / th) if wide else round(width * th / tw)
    a, b = _subject_span(profile, cfg["subject_share"])
    a, b = a * length, b * length
    shift = (b - a) - win
    shift = min(max(shift, cfg["travel"]["min_shift"] * win), cfg["travel"]["max_shift"] * win, length - win)
    centre = (a + b) / 2
    start = min(max(centre - (win + shift) / 2, 0.0), length - win - shift)
    s, e = round(start), round(start + shift)
    if wide:
        return (s, 0, s + win, height), (e, 0, e + win, height)
    return (0, s, width, s + win), (0, e, width, e + win)


def travel(path: Path, dest: Path, end_dest: Path, aspect: str = "9:16") -> ModelKeyframe:
    """Two real windows of one photograph, for a first-and-last-frame model."""
    image = _open(path)
    tw, th = target_size(aspect)
    wide = image.width / image.height >= tw / th
    first, last = travel_windows(image.width, image.height, aspect, _profile(path, wide))
    enlargement = max(tw / (first[2] - first[0]), th / (first[3] - first[1]))
    _save(image.resize((tw, th), Image.LANCZOS, box=first), dest)
    _save(image.resize((tw, th), Image.LANCZOS, box=last), end_dest)
    shift = (last[0] - first[0]) if wide else (last[1] - first[1])
    notes = [f"travel {first} to {last} of {image.width}x{image.height} ({shift} px), enlarged {enlargement:.2f}x"]
    return ModelKeyframe(mode="travel", path=dest, end_path=end_dest, width=tw, height=th,
                         enlargement=round(enlargement, 3), windows=[first, last], notes=notes)


def extend_plan(width: int, height: int, aspect: str = "9:16") -> dict:
    """Where the whole photograph sits in the model's frame and what is left to fill."""
    tw, th = target_size(aspect)
    scale = min(tw / width, th / height)
    bw, bh = min(tw, round(width * scale)), min(th, round(height * scale))
    left, top = (tw - bw) // 2, (th - bh) // 2
    return {"size": (tw, th), "scale": scale, "band_size": (bw, bh), "left": left, "top": top,
            "expand": {"expand_top": top, "expand_bottom": th - bh - top,
                       "expand_left": left, "expand_right": tw - bw - left}}


def outpaint_price(in_size: tuple[int, int], out_size: tuple[int, int]) -> float:
    """USD for one outpaint, by the model page's rule (keyframe.yaml)."""
    import math

    cfg = _cfg()["outpaint"]
    mp_in = in_size[0] * in_size[1] / 1e6
    mp_out = out_size[0] * out_size[1] / 1e6
    extra = max(0, math.ceil(mp_in + mp_out - 1.0 - 1e-9))
    return round(cfg["price_first_mp"] + cfg["price_extra_mp"] * extra, 4)


def _band_offset(answer: np.ndarray, band: np.ndarray, left: int, top: int, search: int) -> tuple[int, int, float]:
    """(dx, dy, mean difference) of the band's best placement within `search` px of where it was put."""
    bh, bw = band.shape[:2]
    best = (0, 0, float("inf"))
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            y, x = top + dy, left + dx
            if y < 0 or x < 0 or y + bh > answer.shape[0] or x + bw > answer.shape[1]:
                continue
            diff = float(np.abs(answer[y:y + bh, x:x + bw] - band).mean())
            if diff < best[2]:
                best = (dx, dy, diff)
    return best


def extend(path: Path, dest: Path, outpaint: Outpaint, aspect: str = "9:16") -> ModelKeyframe:
    """The whole photograph in the middle, the rest filled by `outpaint`, the photograph's own pixels
    put back over the middle. Raises VideoGenerationError when the outpaint moved or changed the
    photograph by more than `band_near_mean`."""
    cfg = _cfg()["outpaint"]
    image = _open(path)
    plan = extend_plan(image.width, image.height, aspect)
    (tw, th), (bw, bh) = plan["size"], plan["band_size"]
    band = image.resize((bw, bh), Image.LANCZOS)
    expand = plan["expand"]
    if not any(expand.values()):
        _save(band, dest)
        return ModelKeyframe(mode="extend", path=dest, width=tw, height=th, enlargement=round(plan["scale"], 3),
                             band=(0, 0, tw, th), notes=["already the video's shape: nothing to fill"])
    if max(expand.values()) > cfg["max_expand_px"]:
        raise VideoGenerationError(f"the outpaint can add at most {cfg['max_expand_px']} px a side")
    answer = outpaint(band, dict(expand)).convert("RGB")
    notes = []
    if answer.size != (tw, th):
        notes.append(f"the outpaint answered {answer.size[0]}x{answer.size[1]}, not {tw}x{th}; resized")
        answer = answer.resize((tw, th), Image.LANCZOS)
    got = np.asarray(answer, dtype=np.float32)
    want = np.asarray(band, dtype=np.float32)
    dx, dy, raw = _band_offset(got, want, plan["left"], plan["top"], int(cfg["band_search_px"]))
    check = {"mean_diff_raw": round(raw, 3), "offset": [dx, dy],
             "verdict": "exact" if raw <= cfg["band_exact_mean"] else "near" if raw <= cfg["band_near_mean"] else "changed"}
    if check["verdict"] == "changed":
        raise VideoGenerationError(f"the outpaint changed the photograph (mean difference {raw:.1f} in the "
                                   f"middle band, limit {cfg['band_near_mean']}); keyframe refused")
    x, y = plan["left"] + dx, plan["top"] + dy
    answer.paste(band, (x, y))
    after = np.asarray(answer, dtype=np.float32)[y:y + bh, x:x + bw]
    check["mean_diff_after"] = round(float(np.abs(after - want).mean()), 3)
    _save(answer, dest)
    notes.append(f"extend: {image.width}x{image.height} scaled {plan['scale']:.3f} to {bw}x{bh} at ({x}, {y}); "
                 f"filled {expand}; band {check['verdict']} (mean difference {raw:.2f} before the pixels were put back)")
    return ModelKeyframe(mode="extend", path=dest, width=tw, height=th, enlargement=round(plan["scale"], 3),
                         generated=True, band=(x, y, x + bw, y + bh), band_check=check,
                         cost_usd=outpaint_price((bw, bh), (tw, th)), notes=notes)
