"""Does a clip from an image-to-video model still show the photograph?

A model asked for camera motion can also move a wall, add a floor or put a
person in a lobby. Two measurements, local and free (numpy and scipy):

  first frame    the clip's first frame against the keyframe it was given:
                 SSIM (structure) and a perceptual hash (the picture as a whole).
                 A model that redraws the start has already left the photograph.
  drift          every half second, the frame is matched to the reference (the
                 keyframe, or the wide photograph for a travel shot) by searching
                 the window of it the camera is looking at (zoom and position), and
                 the two are compared by edge correlation in blocks. Camera motion
                 keeps edges where the matched window has them; a reshaped building
                 or an object that is not in the photograph does not.

For an extended keyframe only the photograph's own band is scored: the filled
areas were generated and are not evidence of anything. Areas a frame shows that
are outside the reference (a pull-back past its edge) are not scored either.

Thresholds are provisional (configs/video/evaluation.yaml, motion_truth); they
were set on synthetic cases (tests/video/test_motion_check.py) and the Citrine
photographs, not on model clips yet. Edge correlation misses small additions
(a person under about a tenth of the frame's height), which is why the prompt
carries the rule too.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy import fft, ndimage

from src.core.config import load_yaml_config
from src.core.exceptions import RenderingError
from src.rendering import ffmpeg

logger = logging.getLogger(__name__)

CONFIG_FILE = "video/evaluation.yaml"


@dataclass
class FrameTruth:
    t: float
    zoom: float                       # 1.0 = the whole reference window; above = closer
    x: float                          # the matched window's centre, as shares of the reference
    y: float
    mean: float                       # block edge correlation: mean, tenth percentile, worst
    p10: float
    worst: float
    below: float                      # share of blocks under motion_truth.drift.below_at
    blocks: int
    verdict: str = "pass"


@dataclass
class ClipTruth:
    first_ssim: float
    first_phash: int
    frames: list[FrameTruth]
    score: float
    verdict: str
    region: str                       # "whole" or "band"
    reasons: list[str] = field(default_factory=list)

    def at(self, t: float) -> Optional[FrameTruth]:
        return min(self.frames, key=lambda f: abs(f.t - t)) if self.frames else None


def _cfg() -> dict:
    return load_yaml_config(CONFIG_FILE)["motion_truth"]


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------

def read_frames(path: Path, width: int, height: int) -> tuple[np.ndarray, float]:
    """Every frame of a clip as RGB uint8, (n, height, width, 3), and its frame rate."""
    info = ffmpeg.probe(path)
    args = [ffmpeg.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-an",
            "-vf", f"scale={width}:{height}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    result = subprocess.run(args, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RenderingError(f"ffmpeg could not read the frames of {Path(path).name}: "
                             f"{result.stderr.decode('utf-8', 'replace')[-300:]}")
    frame = width * height * 3
    count = len(result.stdout) // frame
    frames = np.frombuffer(result.stdout[:count * frame], dtype=np.uint8).reshape(count, height, width, 3)
    return frames, float(info.fps or 25.0)


def gray(image) -> np.ndarray:
    """A float greyscale copy of a PIL image or an RGB array."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    a = np.asarray(image, dtype=np.float32)
    return a if a.ndim == 2 else a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def resize(a: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((width, height), Image.BOX))


def ssim(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None, window: int = 7) -> float:
    """Mean structural similarity of two greyscale pictures of one size (0-255), within `mask`."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = ndimage.uniform_filter(a, window), ndimage.uniform_filter(b, window)
    va = ndimage.uniform_filter(a * a, window) - mu_a ** 2
    vb = ndimage.uniform_filter(b * b, window) - mu_b ** 2
    cov = ndimage.uniform_filter(a * b, window) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s[mask].mean() if mask is not None and mask.any() else s.mean())


def phash(a: np.ndarray) -> int:
    """64-bit perceptual hash: the low frequencies of a 32x32 copy against their median."""
    small = resize(a, 32, 32)
    low = fft.dctn(small, norm="ortho")[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def edges(a: np.ndarray) -> np.ndarray:
    smooth = ndimage.gaussian_filter(a, 1.0)
    return np.hypot(ndimage.sobel(smooth, axis=1), ndimage.sobel(smooth, axis=0))


# ---------------------------------------------------------------------------
# Matching a frame to the reference
# ---------------------------------------------------------------------------

@dataclass
class _Reference:
    image: np.ndarray                 # grey, scaled so its largest frame-shaped window is OVERSAMPLE x the frame
    mask: np.ndarray                  # 1 where the reference is the photograph's own pixels (same scale)
    coarse: np.ndarray                # edges at half the frame's scale, for the first search
    coarse_mask: np.ndarray
    base_w: float                     # that window, in `image` pixels
    base_h: float


OVERSAMPLE = 2


def _reference(ref: np.ndarray, mask: Optional[np.ndarray], fw: int, fh: int) -> _Reference:
    rh, rw = ref.shape
    fit = min(rw / fw, rh / fh)                       # reference pixels per frame pixel at zoom 1
    m = np.ones_like(ref, dtype=np.float32) if mask is None else mask.astype(np.float32)

    def at(per_frame_px: float) -> tuple[np.ndarray, np.ndarray]:
        w, h = max(2, round(rw / fit * per_frame_px)), max(2, round(rh / fit * per_frame_px))
        return resize(ref, w, h), (resize(m, w, h) > 0.5).astype(np.float32)

    image, image_mask = at(OVERSAMPLE)
    small, small_mask = at(0.5)
    return _Reference(image, image_mask, edges(small), small_mask, float(fw * OVERSAMPLE), float(fh * OVERSAMPLE))


def _grid(x0: float, y0: float, ww: float, wh: float, w: int, h: int):
    xs = x0 + (np.arange(w) + 0.5) * ww / w - 0.5
    ys = y0 + (np.arange(h) + 0.5) * wh / h - 0.5
    return np.meshgrid(ys, xs, indexing="ij")


def _warp(ref: _Reference, x0: float, y0: float, z: float, fw: int, fh: int):
    """The reference window at (x0, y0, zoom z) as the frame would show it: edges and mask, (fh, fw)."""
    ww, wh = ref.base_w / z, ref.base_h / z
    gy, gx = _grid(x0, y0, ww, wh, fw * 2, fh * 2)
    img = ndimage.map_coordinates(ref.image, [gy, gx], order=1, mode="nearest")
    m = ndimage.map_coordinates(ref.mask, [gy, gx], order=0, mode="constant", cval=0.0)
    img = img.reshape(fh, 2, fw, 2).mean(axis=(1, 3))
    m = m.reshape(fh, 2, fw, 2).mean(axis=(1, 3))
    return edges(img), (m > 0.99).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    sel = m > 0.5
    if sel.sum() < 16:
        return -1.0
    a, b = a[sel] - a[sel].mean(), b[sel] - b[sel].mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 1e-9 else 0.0


def _fit_score(frame_edges: np.ndarray, e: np.ndarray, m: np.ndarray) -> float:
    """Correlation, discounted when little of the frame falls on the reference."""
    return _ncc(frame_edges, e, m) * min(1.0, float(m.mean()) / 0.5)


def align(ref: _Reference, frame: np.ndarray, cfg: dict,
          seed: Optional[tuple[float, float, float]] = None) -> tuple[float, float, float]:
    """(zoom, left, top), in `ref.image` pixels, of the reference window that best explains the frame.

    A coarse search on edges at half size keeps its best few matches that are not neighbours; each is
    refined on the warped picture, and so is `seed` (the previous frame's match: the camera moves
    continuously). Several starts, because a facade of evenly spaced windows has false matches one
    column off that a single coarse best falls into and refinement cannot climb out of."""
    fh, fw = frame.shape
    search = cfg["search"]
    small = edges(resize(frame, fw // 2, fh // 2))
    sh, sw = small.shape
    ch, cw = ref.coarse.shape
    k = cw / ref.image.shape[1]                     # coarse pixels per reference pixel
    scored: list[tuple[float, tuple[float, float, float]]] = []

    def coarse(z: float, x0: float, y0: float) -> float:
        ww, wh = ref.base_w * k / z, ref.base_h * k / z
        gy, gx = _grid(x0, y0, ww, wh, sw, sh)
        e = ndimage.map_coordinates(ref.coarse, [gy, gx], order=1, mode="constant", cval=0.0)
        m = ndimage.map_coordinates(ref.coarse_mask, [gy, gx], order=0, mode="constant", cval=0.0)
        return _fit_score(small, e, m)

    for z in np.arange(search["min_zoom"], search["max_zoom"] + 1e-6, search["zoom_step"]):
        ww, wh = ref.base_w * k / z, ref.base_h * k / z
        xs = np.arange(min(0.0, cw - ww), max(0.0, cw - ww) + 1e-6, max(1.0, search["pos_step"] * cw))
        ys = np.arange(min(0.0, ch - wh), max(0.0, ch - wh) + 1e-6, max(1.0, search["pos_step"] * ch))
        for y0 in ys:
            for x0 in xs:
                scored.append((coarse(float(z), float(x0), float(y0)), (float(z), float(x0) / k, float(y0) / k)))
    local: list[tuple[float, tuple[float, float, float]]] = []
    if seed is not None:
        # around the previous frame's match, finer: the camera has moved a little since
        reach, near_step = search.get("seed_reach", 0.12), search.get("seed_step", 0.01)
        for z in np.arange(seed[0] - search["zoom_step"], seed[0] + search["zoom_step"] + 1e-6,
                           search["zoom_step"] / 2):
            for dy in np.arange(-reach * ch, reach * ch + 1e-6, max(1.0, near_step * ch)):
                for dx in np.arange(-reach * cw, reach * cw + 1e-6, max(1.0, near_step * cw)):
                    x0, y0 = seed[1] * k + dx, seed[2] * k + dy
                    local.append((coarse(float(z), float(x0), float(y0)), (float(z), float(x0) / k, float(y0) / k)))
    rh, rw = ref.image.shape
    step_x, step_y = search["pos_step"] * rw, search["pos_step"] * rh

    def near(a, b) -> bool:
        return (abs(a[0] - b[0]) <= search["zoom_step"] * 1.5 and abs(a[1] - b[1]) <= step_x * 1.5
                and abs(a[2] - b[2]) <= step_y * 1.5)

    def best_of(found, count: int) -> list[tuple[float, float, float]]:
        picked: list[tuple[float, float, float]] = []
        for _, cand in sorted(found, key=lambda item: -item[0]):
            if not any(near(cand, s) for s in picked):
                picked.append(cand)
            if len(picked) == count:
                break
        return picked

    starts = best_of(scored, int(search.get("starts", 3))) + best_of(local, int(search.get("starts", 3)))
    if seed is not None:
        starts.append(seed)

    # matched on blurred edges: the correlation of sharp edges is a peak a pixel wide, narrower than
    # the refinement's steps, so a start a few pixels off would never climb to it
    blur = float(search.get("align_blur", 1.5))
    fe = ndimage.gaussian_filter(edges(frame), blur)

    def fit(x0: float, y0: float, z: float) -> float:
        e, m = _warp(ref, x0, y0, z, fw, fh)
        return _fit_score(fe, ndimage.gaussian_filter(e, blur), m)

    def refine(start, rounds, dz, dx, dy):
        z, x0, y0 = start
        top = fit(x0, y0, z)
        for _ in range(rounds):
            here = (z, x0, y0)
            for zz in (z - dz / 2, z, z + dz / 2):
                for yy in (y0 - dy / 2, y0, y0 + dy / 2):
                    for xx in (x0 - dx / 2, x0, x0 + dx / 2):
                        score = fit(xx, yy, zz)
                        if score > top:
                            here, top = (float(zz), float(xx), float(yy)), score
            z, x0, y0 = here
            dz, dx, dy = dz / 2, dx / 2, dy / 2
        return top, (z, x0, y0)

    first = [refine(s, 1, search["zoom_step"], step_x, step_y) for s in starts]
    _, best = max(first, key=lambda item: item[0])
    return refine(best, 3, search["zoom_step"] / 2, step_x / 2, step_y / 2)[1]


def _blocks(frame_edges: np.ndarray, ref_edges: np.ndarray, mask: np.ndarray, cfg: dict) -> list[float]:
    rows, cols = cfg["blocks"]["rows"], cfg["blocks"]["cols"]
    fh, fw = frame_edges.shape
    inside = ref_edges[mask > 0.5]
    floor = cfg["blocks"]["min_edge"] * (float(inside.mean()) if inside.size else 0.0)
    scores = []
    for r in range(rows):
        for c in range(cols):
            sl = (slice(r * fh // rows, (r + 1) * fh // rows), slice(c * fw // cols, (c + 1) * fw // cols))
            m = mask[sl]
            if m.mean() < cfg["blocks"]["min_valid"] or ref_edges[sl].mean() < floor:
                continue
            scores.append(_ncc(frame_edges[sl], ref_edges[sl], m))
    return scores


def _grade(value: float, pass_at: float, fail_at: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "pass" if value >= pass_at else "fail" if value < fail_at else "warn"
    return "pass" if value <= pass_at else "fail" if value > fail_at else "warn"


_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def frame_truth(frame: np.ndarray, ref: _Reference, t: float, cfg: dict,
                seed: Optional[tuple[float, float, float]] = None) -> tuple[FrameTruth, tuple[float, float, float]]:
    fh, fw = frame.shape
    z, x0, y0 = align(ref, frame, cfg, seed)
    e, m = _warp(ref, x0, y0, z, fw, fh)
    scores = np.asarray(_blocks(edges(frame), e, m, cfg) or [0.0])
    d = cfg["drift"]
    below = float((scores < d["below_at"]).mean())
    rh, rw = ref.image.shape
    ft = FrameTruth(t=round(t, 2), zoom=round(z, 3), x=round((x0 + ref.base_w / z / 2) / rw, 3),
                    y=round((y0 + ref.base_h / z / 2) / rh, 3), mean=round(float(scores.mean()), 3),
                    p10=round(float(np.percentile(scores, 10)), 3), worst=round(float(scores.min()), 3),
                    below=round(below, 3), blocks=int(len(scores)))
    grades = [_grade(ft.mean, d["mean_pass"], d["mean_fail"]), _grade(ft.p10, d["p10_pass"], d["p10_fail"]),
              _grade(ft.below, d["below_pass"], d["below_fail"], higher_is_better=False)]
    ft.verdict = max(grades, key=_ORDER.get)
    return ft, (z, x0, y0)


# ---------------------------------------------------------------------------
# A whole clip
# ---------------------------------------------------------------------------

def check_frames(frames: list[np.ndarray], times: list[float], keyframe: np.ndarray,
                 reference: Optional[np.ndarray] = None, band: Optional[tuple[int, int, int, int]] = None,
                 start_box: Optional[tuple[int, int, int, int]] = None) -> ClipTruth:
    """The measurements on frames already read (RGB or grey arrays). `keyframe` is what the model was
    given; `reference` the picture drift is measured against (the keyframe when None); `band` the box
    of the keyframe that is the photograph's own pixels (extend), in keyframe pixels; `start_box` where
    the keyframe sits in the reference, in reference pixels, when it is known (a travel keyframe)."""
    cfg = _cfg()
    work_long = int(cfg["work_long_px"])
    kf = gray(keyframe)
    kh, kw = kf.shape
    fw, fh = (round(work_long * kw / kh), work_long) if kh >= kw else (work_long, round(work_long * kh / kw))
    mask = None
    if band is not None:
        mask = np.zeros((kh, kw), bool)
        mask[band[1]:band[3], band[0]:band[2]] = True
    # first frame: at twice the working size, where SSIM still sees texture
    first = resize(gray(frames[0]), fw * 2, fh * 2)
    kf2 = resize(kf, fw * 2, fh * 2)
    m2 = None if mask is None else resize(mask.astype(np.float32), fw * 2, fh * 2) > 0.5
    first_ssim = ssim(first, kf2, m2)
    if band is not None:
        sx, sy = fw * 2 / kw, fh * 2 / kh
        b = (round(band[0] * sx), round(band[1] * sy), round(band[2] * sx), round(band[3] * sy))
        hash_a, hash_b = phash(first[b[1]:b[3], b[0]:b[2]]), phash(kf2[b[1]:b[3], b[0]:b[2]])
    else:
        hash_a, hash_b = phash(first), phash(kf2)
    distance = hamming(hash_a, hash_b)

    if reference is None:
        ref = _reference(kf, mask, fw, fh)
    else:
        ref = _reference(gray(reference), None, fw, fh)
    seed = None
    if reference is not None and start_box is not None:
        per = ref.image.shape[1] / np.asarray(reference).shape[1]
        seed = (ref.base_w / ((start_box[2] - start_box[0]) * per), start_box[0] * per, start_box[1] * per)
    results = []
    for f, t in zip(frames, times):
        ft, seed = frame_truth(resize(gray(f), fw, fh), ref, t, cfg, seed)
        results.append(ft)

    ff = cfg["first_frame"]
    reasons, grades = [], []
    g = _grade(first_ssim, ff["ssim_pass"], ff["ssim_fail"])
    grades.append(g)
    if g != "pass":
        reasons.append(f"first frame SSIM {first_ssim:.2f} against the keyframe ({g})")
    g = _grade(distance, ff["phash_pass"], ff["phash_fail"], higher_is_better=False)
    grades.append(g)
    if g != "pass":
        reasons.append(f"first frame hash {distance} bits from the keyframe ({g})")
    for ft in results:
        grades.append(ft.verdict)
        if ft.verdict != "pass":
            reasons.append(f"{ft.t:.1f} s: edges mean {ft.mean:.2f}, p10 {ft.p10:.2f}, "
                           f"{ft.below:.0%} of blocks under {cfg['drift']['below_at']} ({ft.verdict})")
    w = cfg["score"]
    drift = min((ft.mean for ft in results), default=0.0)
    score = round(100 * (w["first_frame"] * max(0.0, first_ssim) + w["drift"] * max(0.0, drift)), 1)
    return ClipTruth(first_ssim=round(first_ssim, 3), first_phash=distance, frames=results, score=score,
                     verdict=max(grades, key=_ORDER.get), region="whole" if band is None else "band",
                     reasons=reasons)


def sample_times(duration_s: float, every_s: float) -> list[float]:
    times = list(np.arange(every_s, max(every_s, duration_s - 1e-6), every_s))
    return [round(float(t), 3) for t in times] + [round(float(duration_s), 3)]


def check_clip(clip_path: Path, keyframe_path: Path, *, reference_path: Optional[Path] = None,
               band: Optional[tuple[int, int, int, int]] = None, until_s: Optional[float] = None,
               start_box: Optional[tuple[int, int, int, int]] = None) -> ClipTruth:
    """check_frames on a clip file: the first frame and one every motion_truth.sample_every_s, up to
    `until_s` (the part that will be used) or the end."""
    cfg = _cfg()
    with Image.open(keyframe_path) as handle:
        keyframe = np.asarray(handle.convert("RGB"))
    kh, kw = keyframe.shape[:2]
    long_side = int(cfg["work_long_px"]) * 2
    w, h = (round(long_side * kw / kh), long_side) if kh >= kw else (long_side, round(long_side * kh / kw))
    frames, fps = read_frames(clip_path, w - w % 2, h - h % 2)
    if len(frames) == 0:
        raise RenderingError(f"{Path(clip_path).name} has no frames")
    end = min(len(frames) - 1, int(round((until_s or len(frames) / fps) * fps)) - 1)
    times = sample_times(end / fps, float(cfg["sample_every_s"]))
    picked = [frames[0]] + [frames[min(end, int(round(t * fps)))] for t in times]
    reference = None
    if reference_path is not None:
        with Image.open(reference_path) as handle:
            reference = np.asarray(handle.convert("RGB"))
    return check_frames(picked, [0.0] + times, keyframe, reference, band, start_box)
