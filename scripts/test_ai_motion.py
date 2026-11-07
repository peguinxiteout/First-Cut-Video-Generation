#!/usr/bin/env python
"""A 9:16 image-to-video test on fal: can a photograph of any shape become a
full-frame tall moving shot that still shows the photograph, and can model
transitions join the shots into one reel?

    python scripts/test_ai_motion.py                 # the itemised estimate; nothing is sent
    python scripts/test_ai_motion.py --spend         # make the calls (FAL_KEY in the environment)

For three photographs of a property (an exterior for the hook, an amenity, a room):

  keyframes      crop, extend (outpaint on fal) and, for one exterior, travel
                 (agents/video/keyframe.py), with the mode each photograph is recommended
  clips          MiniMax H3 Max (768P, prompt expansion off, push-in) from the crop and the
                 extend keyframe of each; Kling 2.5 Turbo Pro from the hook's crop; one H3
                 travel clip (first and last frame) across the hook's photograph (every Citrine
                 exterior is 1.89:1, so "widest" is the subject: two towers a crop cuts); pan-and-zoom
                 from each photograph, made here, for comparison
  transitions    H3 first-and-last frame: the hook's last frame to the court's first, and the
                 court's last to the living room's first, at the shortest length H3 takes
  truth          every clip against its keyframe and photograph (agents/video/motion_check.py)
  outputs        data/generated_videos/motion_test_<property>/: contact_sheet.png, transitions.png, reel.mp4
                 (1080x1920, silent: hook, transition, court, transition, living; per shot the
                 crop or extend clip with the higher truth score), report.json

Every paid call is listed with its price before anything is sent, the run stops before a
call that would take the total over --cap (at most $4), and nothing is sent without
--spend. Answers are cached by what was sent (data/video_cache/fal_motion/), so a re-run
costs nothing and its estimate says so. The key is read from FAL_KEY in the environment
only and never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MLFLOW_ENABLED", "false")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.agents.video import keyframe as keyframes  # noqa: E402
from src.agents.video import motion_check, motion_prompt  # noqa: E402
from src.core.config import load_yaml_config, settings  # noqa: E402
from src.models.video_gen import ImageToVideoRequest, VideoClip, selector  # noqa: E402
from src.models.video_gen.base import (  # noqa: E402
    VideoGenerationAuthError, VideoGenerationError, VideoGenerationQuotaError,
)
from src.models.video_gen.fal import QUEUE_URL, FalClient, FalOutpainter, FalVideoProvider  # noqa: E402
from src.models.video_gen.kenburns import KenBurnsProvider  # noqa: E402
from src.rendering import ffmpeg  # noqa: E402

logger = logging.getLogger("test_ai_motion")

HARD_CAP_USD = 4.0
FPS = 25
REEL_SIZE = (1080, 1920)
H3, KLING = "minimax_h3_max", "kling_25_turbo_pro"
SHOTS = (("hook", "exterior"), ("court", "amenity"), ("living", "interior"))
DEFAULT_PHOTOS = {"hook": "tower_elevation_view_02.webp", "court": "multiple_court_view.webp",
                  "living": "living.webp"}
TRAVEL_CANDIDATES = ("tower_elevation_view_02.webp", "backside_elevation_view_01.webp",
                     "aerial_view_revise.webp", "pool_to_elevation_view.webp", "tower_b_dropoff_view_02.webp")


class CapReached(Exception):
    """The next call would take the spend over the cap."""


class NotMade(Exception):
    """A dry run reached a call that has not been made: what depends on it is priced, not made."""


# ---------------------------------------------------------------------------
# The calls, priced, cached and counted
# ---------------------------------------------------------------------------

@dataclass
class Call:
    label: str
    endpoint: str
    asked: str                         # what is asked for, in words ("3.0 s", "720x1280")
    price: float                       # the estimate, USD
    state: str                         # cached | to pay | after earlier calls | made | failed | stopped
    cost: float = 0.0                  # what it cost in this run (0 when cached)
    made_cost: float = 0.0             # what the answer cost when it was made
    billed_s: Optional[float] = None
    request_id: str = ""
    note: str = ""


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache_key(endpoint: str, payload: dict) -> str:
    """What was sent, with each data URI replaced by the digest of its bytes."""
    def plain(value):
        if isinstance(value, str) and value.startswith("data:"):
            return "sha256:" + _digest_bytes(value.encode("ascii"))
        return value
    body = {k: plain(v) for k, v in sorted(payload.items())}
    return _digest_bytes(json.dumps({"endpoint": endpoint, "input": body}, sort_keys=True).encode("utf-8"))[:24]


class Ledger:
    def __init__(self, cache_dir: Path, cap: float, spend: bool):
        self.cache_dir, self.cap, self.spend = cache_dir, cap, spend
        self.calls: list[Call] = []
        self.spent = 0.0

    @property
    def to_pay(self) -> float:
        return round(sum(c.price for c in self.calls if c.state in ("to pay", "after earlier calls")), 4)

    def pending(self, label: str, endpoint: str, asked: str, price: float, why: str = "") -> None:
        self.calls.append(Call(label, endpoint, asked, price, "after earlier calls", note=why))

    def _admit(self, call: Call) -> None:
        if self.spent + call.price > self.cap + 1e-9:
            call.state = "stopped"
            raise CapReached(f"{call.label} (${call.price:.3f}) would take the spend from ${self.spent:.3f} "
                             f"over the cap of ${self.cap:.2f}")

    def video(self, label: str, provider: FalVideoProvider, request: ImageToVideoRequest, dest: Path
              ) -> Optional[VideoClip]:
        key = cache_key(provider.endpoint, provider.payload(request))
        video, meta = self.cache_dir / f"{key}.mp4", self.cache_dir / f"{key}.json"
        price = provider.estimate_call(request.duration_s) or 0.0
        asked = f"{provider.duration_value(request.duration_s)} s" + (" first+last" if request.end_image_path else "")
        call = Call(label, provider.endpoint, asked, price, "to pay")
        self.calls.append(call)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if video.exists() and meta.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            call.state, call.price, call.billed_s = "cached", 0.0, data.get("billed_s")
            call.made_cost = float(data.get("cost_usd", 0.0))
            call.request_id, call.note = data.get("request_id", ""), f"paid ${call.made_cost:.3f} when made"
            shutil.copyfile(video, dest)
            return VideoClip(path=dest, provider=data["provider"], model_id=data["model_id"],
                             duration_s=data["duration_s"], width=data["width"], height=data["height"],
                             fps=data["fps"], from_cache=True, meta=data)
        if not self.spend:
            raise NotMade(label)
        self._admit(call)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            clip = provider.generate(request, video)
        except (VideoGenerationAuthError, VideoGenerationQuotaError):
            call.state = "failed"
            raise
        except VideoGenerationError as exc:
            call.state, call.note = "failed", str(exc)[:200]
            logger.warning("%s failed: %s", label, exc)
            return None
        call.state, call.cost, call.billed_s = "made", clip.cost_usd, clip.meta.get("billed_s")
        call.made_cost = clip.cost_usd
        call.request_id = clip.meta.get("request_id", "")
        self.spent = round(self.spent + clip.cost_usd, 4)
        meta.write_text(json.dumps({**clip.meta, "provider": clip.provider, "model_id": clip.model_id,
                                    "duration_s": clip.duration_s, "width": clip.width, "height": clip.height,
                                    "fps": clip.fps, "cost_usd": clip.cost_usd, "label": label,
                                    "made": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
        shutil.copyfile(video, dest)
        clip.path = dest
        return clip

    def outpainter(self, label: str, painter: FalOutpainter) -> Callable[[Image.Image, dict], Image.Image]:
        def outpaint(image: Image.Image, expand: dict) -> Image.Image:
            key = cache_key(painter.endpoint, painter.payload(image, expand))
            png, meta = self.cache_dir / f"{key}.png", self.cache_dir / f"{key}.json"
            out_size = (image.width + expand["expand_left"] + expand["expand_right"],
                        image.height + expand["expand_top"] + expand["expand_bottom"])
            price = keyframes.outpaint_price(image.size, out_size)
            call = Call(label, painter.endpoint, f"{image.width}x{image.height} to {out_size[0]}x{out_size[1]}",
                        price, "to pay")
            self.calls.append(call)
            if png.exists() and meta.exists():
                data = json.loads(meta.read_text(encoding="utf-8"))
                call.state, call.price, call.request_id = "cached", 0.0, data.get("request_id", "")
                call.made_cost = float(data.get("cost_usd", 0.0))
                call.note = f"paid ${call.made_cost:.3f} when made"
                with Image.open(png) as handle:
                    return handle.convert("RGB")
            if not self.spend:
                raise NotMade(label)
            self._admit(call)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                answer = painter(image, expand)
            except VideoGenerationError:
                call.state = "failed"
                raise
            call.state, call.cost, call.made_cost, call.request_id = "made", price, price, painter.last_request_id
            self.spent = round(self.spent + price, 4)
            answer.save(png, "PNG")
            meta.write_text(json.dumps({"request_id": painter.last_request_id, "cost_usd": price, "label": label,
                                        "endpoint": painter.endpoint, "size": list(answer.size),
                                        "made": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
            return answer
        return outpaint


# ---------------------------------------------------------------------------
# Frames and clips, locally
# ---------------------------------------------------------------------------

def frame_at(clip: Path, t: float, dest: Path, size: tuple[int, int]) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(["-ss", f"{max(0.0, t):.3f}", "-i", str(clip), "-frames:v", "1",
                "-vf", f"scale={size[0]}:{size[1]}:flags=lanczos", str(dest)])
    return dest


def last_frame(clip: Path, until_s: float, dest: Path, size: tuple[int, int]) -> Path:
    """The frame on screen at the end of the first `until_s` of a clip (where the reel cuts it)."""
    info = ffmpeg.probe(clip)
    t = min(until_s, info.duration_s) - 1.0 / max(1.0, info.fps or FPS)
    frame_at(clip, t, dest.with_suffix(".png"), size)
    with Image.open(dest.with_suffix(".png")) as handle:
        handle.convert("RGB").save(dest, "JPEG", quality=keyframes.JPEG_QUALITY, subsampling=0)
    return dest


def _truth_key(clip: Path, kf: Path, reference: Optional[Path], extra: list) -> str:
    cfg = json.dumps(load_yaml_config(motion_check.CONFIG_FILE)["motion_truth"], sort_keys=True)
    parts = [_digest_bytes(Path(p).read_bytes()) for p in (clip, kf, reference) if p is not None]
    return _digest_bytes(json.dumps([parts, extra, cfg]).encode("utf-8"))[:24]


def truth(clip: Path, kf: Path, cache: Path, *, reference: Optional[Path] = None, band=None,
          start_box=None, until_s: float) -> dict:
    """motion_check on a clip, cached by the clip, the keyframe, the reference and the thresholds."""
    key = _truth_key(clip, kf, reference, [list(band) if band else None,
                                           list(start_box) if start_box else None, until_s])
    path = cache / f"truth_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = motion_check.check_clip(clip, kf, reference_path=reference, band=band, until_s=until_s,
                                     start_box=start_box)
    data = {"score": result.score, "verdict": result.verdict, "first_ssim": result.first_ssim,
            "first_phash": result.first_phash, "region": result.region, "reasons": result.reasons,
            "frames": [vars(f) for f in result.frames]}
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def transition_check(clip: Path, start: Path, end: Path, work: Path) -> dict:
    """A transition joins two pictures: its first frame against the start, its last against the end."""
    info = ffmpeg.probe(clip)
    size = (216, 384)
    a = frame_at(clip, 0.0, work / "t_first.png", size)
    b = frame_at(clip, info.duration_s - 1.0 / max(1.0, info.fps or FPS), work / "t_last.png", size)

    def score(x: Path, y: Path) -> float:
        with Image.open(x) as p, Image.open(y) as q:
            return round(motion_check.ssim(motion_check.gray(p), motion_check.gray(q.convert("RGB").resize(size))), 3)
    return {"first_ssim": score(a, start), "last_ssim": score(b, end), "duration_s": info.duration_s}


def pan_and_zoom(photo: Path, focal: str, seconds: float, dest: Path, size: tuple[int, int]) -> VideoClip:
    request = ImageToVideoRequest(image_path=photo, camera="push_in", duration_s=seconds, width=size[0],
                                  height=size[1], fps=FPS, focal=focal)
    return KenBurnsProvider().generate(request, dest)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

@dataclass
class Shot:
    name: str
    scene: str
    photo: Path
    recommendation: Optional[keyframes.Recommendation] = None
    crop: Optional[keyframes.ModelKeyframe] = None
    extend: Optional[keyframes.ModelKeyframe] = None
    extend_note: str = ""
    clips: dict = field(default_factory=dict)          # column -> VideoClip
    truths: dict = field(default_factory=dict)         # column -> truth dict
    chosen: str = ""


def _providers(api_key: str, queue_url: str) -> tuple[FalVideoProvider, FalVideoProvider, FalClient]:
    catalog = selector.load_catalog()
    client = FalClient(api_key, queue_url=queue_url)
    return (FalVideoProvider(catalog[H3], api_key, client), FalVideoProvider(catalog[KLING], api_key, client),
            client)


def _request(kf: Path, prompt: motion_prompt.MotionPrompt, seconds: float, *, end: Optional[Path] = None,
             camera: str = "push_in") -> ImageToVideoRequest:
    w, h = keyframes.target_size()
    return ImageToVideoRequest(image_path=kf, end_image_path=end, prompt=prompt.prompt,
                               negative_prompt=prompt.negative_prompt, camera=camera, duration_s=seconds,
                               width=w, height=h, fps=FPS)


def pick_travel(images: Path, names: list[str]) -> Optional[Path]:
    """The exterior whose detail a crop keeps least of: the widest subject."""
    found = [(keyframes.recommend(images / n, topic="exterior"), images / n) for n in names if (images / n).is_file()]
    if not found:
        return None
    return min(found, key=lambda item: (-item[0].aspect, item[0].crop_detail))[1]


def run(args, ledger: Ledger, out: Path, h3: FalVideoProvider, kling: FalVideoProvider, painter: FalOutpainter) -> dict:
    images = settings.data_dir_resolved / "properties" / args.property_id / "images"
    keys, clips_dir, work = out / "keyframes", out / "clips", out / "work"
    size = keyframes.target_size()
    shots = [Shot(name, scene, images / getattr(args, name)) for name, scene in SHOTS]
    for shot in shots:
        if not shot.photo.is_file():
            raise SystemExit(f"No photograph {shot.photo}")
    h3_prompt = {s.name: motion_prompt.for_provider(h3.params, "push_in", scene=s.scene) for s in shots}
    out_paint_endpoint = keyframes._cfg()["outpaint"]["endpoint"]

    # keyframes and the shots' clips
    for shot in shots:
        shot.recommendation = keyframes.recommend(shot.photo, topic=shot.scene)
        shot.crop = keyframes.crop(shot.photo, keys / f"{shot.name}_crop.jpg", topic=shot.scene)
        try:
            shot.extend = keyframes.extend(shot.photo, keys / f"{shot.name}_extend.jpg",
                                           ledger.outpainter(f"{shot.name}: outpaint to 9:16", painter))
        except NotMade:
            shot.extend_note = "not made yet"
        except (VideoGenerationAuthError, VideoGenerationQuotaError):
            raise
        except VideoGenerationError as exc:
            shot.extend_note = str(exc)
        if ledger.spend:
            shot.clips["pan_zoom"] = pan_and_zoom(shot.photo, shot.crop.focal, args.shot_s,
                                                  clips_dir / f"{shot.name}_pan_zoom.mp4", size)
        for column, kf in (("crop", shot.crop), ("extend", shot.extend)):
            label = f"{shot.name}: H3 Max from the {column} keyframe"
            if kf is None:
                if shot.extend_note == "not made yet":
                    ledger.pending(label, h3.endpoint, f"{h3.duration_value(args.shot_s)} s",
                                   h3.estimate_call(args.shot_s) or 0.0, "needs the outpaint first")
                continue
            try:
                clip = ledger.video(label, h3, _request(kf.path, h3_prompt[shot.name], args.shot_s),
                                    clips_dir / f"{shot.name}_{column}_h3.mp4")
            except NotMade:
                continue
            if clip is not None:
                shot.clips[column] = clip

    hook = shots[0]
    kling_prompt = motion_prompt.for_provider(kling.params, "push_in", scene=hook.scene)
    try:
        clip = ledger.video("hook: Kling 2.5 Turbo Pro from the crop keyframe (comparison)", kling,
                            _request(hook.crop.path, kling_prompt, args.shot_s), clips_dir / "hook_crop_kling.mp4")
        if clip is not None:
            hook.clips["kling"] = clip
    except NotMade:
        pass

    # travel
    if args.travel == "hook":
        travel_photo = shots[0].photo
    elif args.travel == "auto":
        travel_photo = pick_travel(images, list(TRAVEL_CANDIDATES))
    else:
        travel_photo = images / args.travel
    travel_info: dict = {}
    if travel_photo is not None:
        tk = keyframes.travel(travel_photo, keys / "travel_start.jpg", keys / "travel_end.jpg")
        direction = "pan_right" if tk.windows[1][0] >= tk.windows[0][0] else "pan_left"
        prompt = motion_prompt.for_provider(h3.params, direction, scene="exterior")
        travel_info = {"photo": travel_photo.name, "keyframe": tk, "camera": direction,
                       "recommendation": keyframes.recommend(travel_photo, topic="exterior")}
        try:
            clip = ledger.video(f"travel: H3 Max first+last frame across {travel_photo.name}", h3,
                                _request(tk.path, prompt, args.shot_s, end=tk.end_path, camera=direction),
                                clips_dir / "travel_h3.mp4")
            if clip is not None:
                travel_info["clip"] = clip
        except NotMade:
            pass

    # truth, then the better of crop and extend per shot
    for shot in shots:
        for column, clip in shot.clips.items():
            if column == "extend":
                shot.truths[column] = truth(clip.path, shot.extend.path, ledger.cache_dir, band=shot.extend.band,
                                            until_s=args.shot_s)
            else:
                shot.truths[column] = truth(clip.path, shot.crop.path, ledger.cache_dir, until_s=args.shot_s)
        model = {c: shot.truths[c]["score"] for c in ("crop", "extend") if c in shot.truths}
        shot.chosen = max(model, key=lambda c: (model[c], c == "crop")) if model else ""
    if "clip" in travel_info:
        tk = travel_info["keyframe"]
        travel_info["truth"] = truth(travel_info["clip"].path, tk.path, ledger.cache_dir,
                                     reference=travel_photo, start_box=tk.windows[0], until_s=args.shot_s)

    # transitions
    transition_prompt = motion_prompt.for_provider(h3.params, transition=True)
    transitions = []
    for a, b in ((shots[0], shots[1]), (shots[1], shots[2])):
        label = f"transition: {a.name} to {b.name}, H3 Max first+last frame"
        end_kf = b.extend if b.chosen == "extend" else b.crop
        if not a.chosen or not b.chosen:
            ledger.pending(label, h3.endpoint, f"{h3.duration_value(args.transition_ask_s)} s first+last",
                           h3.estimate_call(args.transition_ask_s) or 0.0, "needs both shots' clips first")
            transitions.append({"from": a.name, "to": b.name})
            continue
        start = last_frame(a.clips[a.chosen].path, args.shot_s, keys / f"{a.name}_last.jpg", size)
        item = {"from": a.name, "to": b.name, "start": start, "end": end_kf.path}
        try:
            clip = ledger.video(label, h3, _request(start, transition_prompt, args.transition_ask_s, end=end_kf.path),
                                clips_dir / f"transition_{a.name}_{b.name}.mp4")
            if clip is not None:
                item["clip"] = clip
                item["check"] = transition_check(clip.path, start, end_kf.path, work)
        except NotMade:
            pass
        transitions.append(item)
    return {"shots": shots, "travel": travel_info, "transitions": transitions}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:        # Pillow before 10.1
        return ImageFont.load_default()


THUMB = (150, 267)
COLUMNS = (("pan_zoom", "Pan-and-zoom (free)"), ("crop", "H3 Max, crop"), ("extend", "H3 Max, extend"),
           ("kling", "Kling 2.5 Turbo Pro, crop"))


def _clip_cell(clip: VideoClip, data: Optional[dict], seconds: float, work: Path, tag: str, cost: float) -> Image.Image:
    gap, text_h = 4, 58
    cell = Image.new("RGB", (THUMB[0] * 3 + gap * 2, THUMB[1] + text_h), "white")
    draw = ImageDraw.Draw(cell)
    small = _font(12)
    used = min(seconds, clip.duration_s)
    for i, t in enumerate((0.0, used / 2, used - 1.0 / FPS)):
        frame = frame_at(clip.path, t, work / f"{tag}_{i}.png", THUMB)
        with Image.open(frame) as im:
            cell.paste(im.convert("RGB"), (i * (THUMB[0] + gap), 0))
        if data:
            ft = min(data["frames"], key=lambda f: abs(f["t"] - t)) if data.get("frames") else None
            label = f"{t:.1f}s" + (f" edges {ft['mean']:.2f}" if ft else "")
            if i == 0:
                label += f" ssim {data['first_ssim']:.2f}"
            draw.rectangle((i * (THUMB[0] + gap), THUMB[1] - 16, i * (THUMB[0] + gap) + THUMB[0], THUMB[1]),
                           fill=(0, 0, 0))
            draw.text((i * (THUMB[0] + gap) + 3, THUMB[1] - 15), label, fill="white", font=small)
    lines = [f"{clip.duration_s:.1f} s made, {used:.1f} s used, ${cost:.3f}"]
    if data:
        lines.insert(0, f"truth {data['score']:.0f} ({data['verdict']}), region {data['region']}")
    for j, line in enumerate(lines):
        draw.text((2, THUMB[1] + 4 + j * 17), line, fill="black", font=_font(14))
    return cell


def contact_sheet(result: dict, ledger: Ledger, seconds: float, dest: Path, work: Path) -> Path:
    shots, travel_info = result["shots"], result["travel"]
    def cost_of(prefix: str) -> float:
        return sum(c.made_cost for c in ledger.calls if c.label.startswith(prefix))
    rows = [(s.name, s) for s in shots] + ([("travel", None)] if travel_info else [])
    label_w, key_w, cell_w, row_h = 220, THUMB[0] + 10, THUMB[0] * 3 + 8 + 16, THUMB[1] + 70
    sheet = Image.new("RGB", (label_w + key_w + cell_w * len(COLUMNS), 40 + row_h * len(rows)), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "Row", fill="black", font=_font(16))
    draw.text((label_w, 10), "Keyframe", fill="black", font=_font(16))
    for i, (_, title) in enumerate(COLUMNS):
        draw.text((label_w + key_w + i * cell_w, 10), title, fill="black", font=_font(16))
    for r, (name, shot) in enumerate(rows):
        y = 40 + r * row_h
        if shot is None:
            tk, rec = travel_info["keyframe"], travel_info["recommendation"]
            text = [f"travel: {travel_info['photo']}", f"{travel_info['camera']}, first+last frame",
                    f"recommended: {rec.mode}", tk.notes[0][:44] if tk.notes else "",
                    "(the H3 Max, crop column)"]
            with Image.open(tk.path) as im:
                sheet.paste(im.convert("RGB").resize(THUMB), (label_w, y))
            if "clip" in travel_info:
                sheet.paste(_clip_cell(travel_info["clip"], travel_info.get("truth"), seconds, work, "travel",
                                       cost_of("travel")), (label_w + key_w + cell_w, y))
        else:
            rec = shot.recommendation
            text = [f"{name}: {shot.photo.name}", f"recommended: {rec.mode}",
                    f"crop keeps {rec.crop_keep:.0%} area, {rec.crop_detail:.0%} detail",
                    f"crop enlarged {shot.crop.enlargement:.2f}x",
                    f"chosen for the reel: {shot.chosen or '-'}"]
            if shot.extend is not None:
                text.append(f"extend band: {shot.extend.band_check.get('verdict', '-')} "
                            f"(diff {shot.extend.band_check.get('mean_diff_raw', '-')})")
            elif shot.extend_note:
                text.append(f"extend: {shot.extend_note[:40]}")
            with Image.open(shot.crop.path) as im:
                sheet.paste(im.convert("RGB").resize(THUMB), (label_w, y))
            for i, (column, _) in enumerate(COLUMNS):
                if column not in shot.clips:
                    continue
                spent = {"pan_zoom": 0.0, "crop": cost_of(f"{name}: H3 Max from the crop"),
                         "extend": cost_of(f"{name}: H3 Max from the extend") + cost_of(f"{name}: outpaint"),
                         "kling": cost_of("hook: Kling")}[column]
                sheet.paste(_clip_cell(shot.clips[column], shot.truths.get(column), seconds, work,
                                       f"{name}_{column}", spent), (label_w + key_w + i * cell_w, y))
        for j, line in enumerate(text):
            draw.text((8, y + j * 18), line, fill="black", font=_font(13))
    sheet.save(dest)
    return dest


def transitions_sheet(result: dict, dest: Path, work: Path) -> Optional[Path]:
    items = [t for t in result["transitions"] if "clip" in t]
    if not items:
        return None
    every = 0.5
    rows = []
    for t in items:
        info = ffmpeg.probe(t["clip"].path)
        times = [round(i * every, 2) for i in range(int(info.duration_s / every) + 1)]
        rows.append((t, times))
    width = max(len(times) + 2 for _, times in rows) * (THUMB[0] + 6) + 10
    sheet = Image.new("RGB", (width, len(rows) * (THUMB[1] + 50) + 10), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for r, (t, times) in enumerate(rows):
        y = 10 + r * (THUMB[1] + 50)
        pictures = [("start", t["start"])] + \
            [(f"{s:.1f}s", frame_at(t["clip"].path, s, work / f"tr{r}_{i}.png", THUMB)) for i, s in enumerate(times)] + \
            [("end", t["end"])]
        for i, (label, path) in enumerate(pictures):
            with Image.open(path) as im:
                sheet.paste(im.convert("RGB").resize(THUMB), (10 + i * (THUMB[0] + 6), y))
            draw.text((12 + i * (THUMB[0] + 6), y + THUMB[1] + 2), label, fill="black", font=_font(13))
        check = t.get("check", {})
        draw.text((12, y + THUMB[1] + 20), f"{t['from']} to {t['to']}: {check.get('duration_s', 0):.1f} s made; "
                  f"first frame SSIM to the start {check.get('first_ssim', 0):.2f}, last frame to the end "
                  f"{check.get('last_ssim', 0):.2f}", fill="black", font=_font(13))
    sheet.save(dest)
    return dest


def reel(result: dict, shot_s: float, transition_s: float, dest: Path) -> Optional[Path]:
    """hook, transition, court, transition, living at 1080x1920, silent. A shot is its first `shot_s`;
    a transition is retimed (not cut) to `transition_s`, so it still starts and ends on its two frames."""
    shots = result["shots"]
    parts: list[tuple[Path, str]] = []
    for i, shot in enumerate(shots):
        column = shot.chosen or "pan_zoom"
        if column not in shot.clips:
            return None
        parts.append((shot.clips[column].path, "shot"))
        if i < len(result["transitions"]):
            t = result["transitions"][i]
            if "clip" in t:
                parts.append((t["clip"].path, "transition"))
    w, h = REEL_SIZE
    fit = f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},setsar=1,fps={FPS}"
    args, chains = [], []
    for i, (path, kind) in enumerate(parts):
        args += ["-i", str(path)]
        if kind == "shot":
            chains.append(f"[{i}:v]trim=0:{shot_s},setpts=PTS-STARTPTS,{fit}[v{i}]")
        else:
            made = ffmpeg.probe(path).duration_s or transition_s
            chains.append(f"[{i}:v]setpts=(PTS-STARTPTS)*{transition_s / made:.5f},{fit},trim=0:{transition_s},"
                          f"setpts=PTS-STARTPTS[v{i}]")
    joined = "".join(f"[v{i}]" for i in range(len(parts)))
    chains.append(f"{joined}concat=n={len(parts)}:v=1:a=0[out]")
    ffmpeg.run([*args, "-filter_complex", ";".join(chains), "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "18",
                "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)], timeout=600)
    return dest


def report(result: dict, ledger: Ledger, args, out: Path, files: dict) -> Path:
    def clip_data(clip: VideoClip) -> dict:
        return {"file": str(clip.path.relative_to(out)), "model": clip.model_id, "seconds": clip.duration_s,
                "size": [clip.width, clip.height], "cost_usd": clip.cost_usd, "cached": clip.from_cache,
                "billed_s": clip.meta.get("billed_s"), "request_id": clip.meta.get("request_id", "")}
    data = {"property": args.property_id, "shot_s": args.shot_s, "transition_s": args.transition_s,
            "spent_usd": ledger.spent, "cap_usd": ledger.cap,
            "calls": [vars(c) for c in ledger.calls],
            "shots": [{"name": s.name, "photo": s.photo.name, "recommendation": vars(s.recommendation),
                       "crop": {"enlargement": s.crop.enlargement, "focal": s.crop.focal, "window": s.crop.windows},
                       "extend": None if s.extend is None else {"band": s.extend.band, "band_check": s.extend.band_check,
                                                                "generated": s.extend.generated, "notes": s.extend.notes},
                       "extend_note": s.extend_note, "chosen": s.chosen,
                       "clips": {c: clip_data(v) for c, v in s.clips.items()}, "truth": s.truths}
                      for s in result["shots"]],
            "travel": {k: (clip_data(v) if isinstance(v, VideoClip) else vars(v) if hasattr(v, "__dict__") else v)
                       for k, v in result["travel"].items() if k != "keyframe"},
            "transitions": [{k: (clip_data(v) if isinstance(v, VideoClip) else str(v) if isinstance(v, Path) else v)
                             for k, v in t.items()} for t in result["transitions"]],
            "files": {k: str(v) for k, v in files.items() if v}}
    path = out / "report.json"
    path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------

def print_estimate(ledger: Ledger, write=print) -> None:
    write("")
    write(f"{'#':>2}  {'call':58s} {'endpoint':48s} {'asked':20s} {'price':>8s}  state")
    for i, c in enumerate(ledger.calls, 1):
        state = c.state + (f" ({c.note})" if c.note and c.state != "to pay" else "")
        write(f"{i:>2}  {c.label[:58]:58s} {c.endpoint[:48]:48s} {c.asked[:20]:20s} ${c.price:7.3f}  {state}")
    write(f"\nTo pay: ${ledger.to_pay:.3f} for {sum(c.state != 'cached' for c in ledger.calls)} call(s); "
          f"{sum(c.state == 'cached' for c in ledger.calls)} cached (free). Cap: ${ledger.cap:.2f}.")
    write("Prices: MiniMax H3 Max 768P $0.08/s of video made (standard rate; $0.056/s until 15 Nov 2026), "
          "a clip runs about 0.7 s over what is asked; Kling 2.5 Turbo Pro $0.35 per 5 s; outpaint "
          "$0.03 first megapixel + $0.015 per extra megapixel in and out.")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--property-id", default="brigade-citrine")
    p.add_argument("--hook", default=DEFAULT_PHOTOS["hook"], help="exterior photograph for the hook")
    p.add_argument("--court", default=DEFAULT_PHOTOS["court"])
    p.add_argument("--living", default=DEFAULT_PHOTOS["living"])
    p.add_argument("--travel", default="hook",
                   help="photograph for the travel clip: hook (the hook's photograph), auto (the exterior a crop "
                        "keeps least of) or a file name")
    p.add_argument("--shot-s", type=float, default=3.0, help="seconds of each shot in the reel (asked of the model)")
    p.add_argument("--transition-s", type=float, default=2.5, help="seconds of each transition in the reel")
    p.add_argument("--transition-ask-s", type=float, default=2.0, help="seconds asked of the model for a transition")
    p.add_argument("--cap", type=float, default=HARD_CAP_USD, help=f"stop before spending more (at most {HARD_CAP_USD})")
    p.add_argument("--spend", action="store_true", help="make the paid calls; without it only the estimate is printed")
    p.add_argument("--out", default="", help="output folder (default data/generated_videos/motion_test_<property>)")
    return p.parse_args(argv)


def main(argv=None, *, queue_url: str = QUEUE_URL, write=print) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cap = min(args.cap, HARD_CAP_USD)
    out = Path(args.out) if args.out else settings.data_dir_resolved / "generated_videos" / f"motion_test_{args.property_id}"
    for sub in ("keyframes", "clips", "work"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cache = settings.data_dir_resolved / "video_cache" / "fal_motion"
    api_key = os.environ.get("FAL_KEY", "").strip()
    h3, kling, client = _providers(api_key, queue_url)
    painter = FalOutpainter(client, keyframes._cfg()["outpaint"]["endpoint"], keyframes._cfg()["outpaint"]["params"])

    estimate = Ledger(cache, cap, spend=False)
    run(args, estimate, out, h3, kling, painter)
    print_estimate(estimate, write)
    if estimate.to_pay > cap + 1e-9:
        write(f"\nAborted: the estimate ${estimate.to_pay:.3f} is over the cap of ${cap:.2f}. Nothing was sent.")
        return 2
    if not args.spend:
        write("\nNothing was sent. Run again with --spend to make these calls.")
        return 0
    if not api_key:
        write("\nFAL_KEY is not set in this shell. Nothing was sent.")
        return 2

    ledger = Ledger(cache, cap, spend=True)
    started = time.monotonic()
    try:
        result = run(args, ledger, out, h3, kling, painter)
    except CapReached as exc:
        write(f"\nStopped: {exc}. Spent ${ledger.spent:.3f}.")
        print_estimate(ledger, write)
        return 3
    except (VideoGenerationAuthError, VideoGenerationQuotaError) as exc:
        write(f"\nStopped: {exc}. Spent ${ledger.spent:.3f}.")
        print_estimate(ledger, write)
        return 3
    files = {"contact_sheet": contact_sheet(result, ledger, args.shot_s, out / "contact_sheet.png", out / "work"),
             "transitions": transitions_sheet(result, out / "transitions.png", out / "work"),
             "reel": reel(result, args.shot_s, args.transition_s, out / "reel.mp4")}
    files["report"] = report(result, ledger, args, out, files)
    print_estimate(ledger, write)
    write(f"\nSpent ${ledger.spent:.3f} in {time.monotonic() - started:.0f} s.")
    for shot in result["shots"]:
        scores = ", ".join(f"{c} {d['score']:.0f} {d['verdict']}" for c, d in shot.truths.items())
        write(f"{shot.name}: recommended {shot.recommendation.mode}; chosen {shot.chosen or 'pan_zoom'}; truth: {scores}")
    for name, path in files.items():
        if path:
            write(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
